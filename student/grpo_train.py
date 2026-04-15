"""GRPO training loop for Countdown.

This script implements the full GRPO train loop used in the assignment:
1) sample rollouts from the current policy
2) compute grouped rewards / advantages
3) run microbatch policy-gradient updates
4) periodically evaluate on validation prompts
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.drgrpo_grader import question_only_reward_fn
from student.sft_vllm_utils import init_vllm, load_policy_into_vllm_instance


def _load_prompt_template() -> str:
    p = Path(__file__).parent / "prompts" / "countdown.prompt"
    return p.read_text(encoding="utf-8")


def _wandb_init(args: argparse.Namespace) -> Any:
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; logging to stdout only. pip install wandb")
        return None

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return wandb


def _extract_countdown_fields(ex: dict[str, Any]) -> tuple[str, str]:
    """Return (question_text, ground_truth) from multiple common field layouts."""
    if "problem" in ex and "answer" in ex:
        return str(ex["problem"]), str(ex["answer"])
    if "question" in ex and "answer" in ex:
        return str(ex["question"]), str(ex["answer"])
    if "nums" in ex and "target" in ex:
        nums = ex["nums"]
        target = ex["target"]
        question = f"Use numbers {nums} exactly once to make target {target}."
        gt = str(ex.get("answer", ex.get("solution", "")))
        return question, gt
    if "input" in ex and "target" in ex:
        question = str(ex["input"])
        gt = str(ex.get("answer", ex.get("target")))
        return question, gt
    raise ValueError(f"Unsupported countdown schema. Keys: {sorted(ex.keys())}")


def _load_countdown_splits(
    dataset_name: str,
    train_split: str,
    val_split: str,
    max_train_samples: int | None,
    max_val_samples: int,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    train_ds = load_dataset(dataset_name, split=train_split)
    val_ds = load_dataset(dataset_name, split=val_split)

    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train_samples))
    if max_val_samples and max_val_samples < len(val_ds):
        val_ds = val_ds.shuffle(seed=seed).select(range(max_val_samples))

    train_questions: list[str] = []
    train_gts: list[str] = []
    for ex in train_ds:
        q, gt = _extract_countdown_fields(ex)
        train_questions.append(q)
        train_gts.append(gt)

    val_questions: list[str] = []
    val_gts: list[str] = []
    for ex in val_ds:
        q, gt = _extract_countdown_fields(ex)
        val_questions.append(q)
        val_gts.append(gt)

    return train_questions, train_gts, val_questions, val_gts


def _tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    full_ids_list: list[list[int]] = []
    for p, o in zip(prompt_ids_list, output_ids_list):
        full = p + o
        if len(full) > max_seq_len:
            full = full[:max_seq_len]
        if len(full) < 2:
            # Keep at least two tokens so shift works.
            full = (full + [tokenizer.eos_token_id, tokenizer.eos_token_id])[:2]
        full_ids_list.append(full)

    batch_size = len(full_ids_list)
    prompt_lens = [min(len(p), len(f)) for p, f in zip(prompt_ids_list, full_ids_list)]
    full_lens = [len(f) for f in full_ids_list]
    max_len = max(full_lens)
    seq_len = max_len - 1

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded_full = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i in range(batch_size):
        ids = full_ids_list[i]
        padded_full[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    input_ids = padded_full[:, :-1].clone()
    labels = padded_full[:, 1:].clone()

    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i in range(batch_size):
        start = max(0, prompt_lens[i] - 1)
        end = max(0, full_lens[i] - 1)
        if start < end:
            response_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def _get_response_log_probs_and_entropy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(input_ids=input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = torch.exp(log_probs_all)
    token_entropy = -(probs * log_probs_all).sum(dim=-1)
    return log_probs, token_entropy


@torch.no_grad()
def _old_policy_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    logits = model(input_ids=input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)
    out = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if was_training:
        model.train()
    return out


def _compute_group_normalized_rewards(
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = []
    raw_format = []
    raw_answer = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        r = question_only_reward_fn(resp, gt)
        raw_rewards.append(float(r["reward"]))
        raw_format.append(float(r["format_reward"]))
        raw_answer.append(float(r["answer_reward"]))

    raw_rewards_t = torch.tensor(raw_rewards, dtype=torch.float32)
    grouped = raw_rewards_t.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std = grouped.std(dim=1, unbiased=True, keepdim=True)
        adv = centered / (std + float(advantage_eps))
    else:
        adv = centered

    metadata = {
        "reward_mean": float(raw_rewards_t.mean().item()),
        "reward_std": float(raw_rewards_t.std(unbiased=False).item()),
        "reward_format_mean": float(torch.tensor(raw_format).mean().item()),
        "reward_answer_mean": float(torch.tensor(raw_answer).mean().item()),
    }
    return adv.reshape(-1), raw_rewards_t, metadata


def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask_f = mask.to(tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        return masked.sum() / mask_f.sum().clamp_min(1.0)
    return masked.sum(dim=dim) / mask_f.sum(dim=dim).clamp_min(1.0)


def _compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return -(raw_rewards * policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        return -(advantages * policy_log_probs), {}

    if old_log_probs is None:
        raise ValueError("old_log_probs required for loss_type='grpo_clip'")
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    lhs = ratio * advantages
    rhs = clipped_ratio * advantages
    per_token_loss = -torch.minimum(lhs, rhs)
    metadata = {
        "clip_fraction": ((ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))).to(torch.float32),
    }
    return per_token_loss, metadata


def _build_countdown_prompts(prompt_template: str, questions: list[str]) -> list[str]:
    return [prompt_template.format(question=q) for q in questions]


def _sample_rollouts_vllm(
    vllm_llm: Any,
    prompts: list[str],
    group_size: int,
    temperature: float,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    from vllm import SamplingParams

    repeated_prompts = []
    for p in prompts:
        repeated_prompts.extend([p] * group_size)

    params = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=["</answer>"],
    )
    outputs = vllm_llm.generate(repeated_prompts, params)
    return [o.outputs[0].text for o in outputs]


def _eval_val_reward(
    vllm_llm: Any,
    prompt_template: str,
    val_questions: list[str],
    val_ground_truths: list[str],
    max_eval_examples: int,
    max_tokens: int,
) -> float:
    from vllm import SamplingParams

    n = min(max_eval_examples, len(val_questions))
    prompts = _build_countdown_prompts(prompt_template, val_questions[:n])
    gts = val_ground_truths[:n]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=["</answer>"])
    outputs = vllm_llm.generate(prompts, params)
    rewards = [question_only_reward_fn(out.outputs[0].text, gt)["reward"] for out, gt in zip(outputs, gts)]
    return float(sum(rewards) / max(1, len(rewards)))


def train_loop(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("GRPO training requires CUDA GPUs (policy + optional vLLM GPU).")

    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    assert args.rollout_batch_size % micro_train_batch_size == 0, (
        "rollout_batch_size must be divisible by micro_train_batch_size"
    )
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    prompt_template = _load_prompt_template()
    train_questions, train_gts, val_questions, val_gts = _load_countdown_splits(
        dataset_name=args.countdown_dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(f"Loaded countdown dataset | train={len(train_questions)} val={len(val_questions)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(torch.device(args.device))
    policy.train()

    optimizer = AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    wb = _wandb_init(args)

    print(f"Initializing vLLM on {args.vllm_device} ...")
    vllm_llm = init_vllm(
        model_id=args.model_id,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    pbar = tqdm(total=args.n_grpo_steps, desc="grpo")
    for step in range(1, args.n_grpo_steps + 1):
        # --- sample prompt mini-set for this rollout batch ---
        idxs = random.sample(range(len(train_questions)), k=n_prompts_per_rollout_batch)
        prompt_batch = [train_questions[i] for i in idxs]
        gt_batch = [train_gts[i] for i in idxs]
        prompt_strs = _build_countdown_prompts(prompt_template, prompt_batch)

        # --- rollout from current policy via vLLM ---
        load_policy_into_vllm_instance(policy, vllm_llm)
        rollout_responses = _sample_rollouts_vllm(
            vllm_llm=vllm_llm,
            prompts=prompt_strs,
            group_size=args.group_size,
            temperature=args.sampling_temperature,
            min_tokens=args.sampling_min_tokens,
            max_tokens=args.sampling_max_tokens,
        )
        repeated_prompts = [p for p in prompt_strs for _ in range(args.group_size)]
        repeated_gts = [g for g in gt_batch for _ in range(args.group_size)]

        # --- tokenize for policy scoring ---
        toks = _tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_length,
        )
        input_ids = toks["input_ids"].to(args.device)
        labels = toks["labels"].to(args.device)
        response_mask = toks["response_mask"].to(args.device)

        # --- rewards/advantages (per response) ---
        advantages_1d, raw_rewards_1d, reward_meta = _compute_group_normalized_rewards(
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        advantages = advantages_1d.to(args.device).unsqueeze(1)
        raw_rewards = raw_rewards_1d.to(args.device).unsqueeze(1)

        # off-policy term if requested
        old_log_probs_all = None
        if args.loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs_all = _old_policy_log_probs(policy, input_ids, labels)

        # --- microbatch train step(s) ---
        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        clip_frac_count = 0

        for mb in range(n_microbatches_per_rollout_batch):
            s = mb * micro_train_batch_size
            e = (mb + 1) * micro_train_batch_size

            mb_input_ids = input_ids[s:e]
            mb_labels = labels[s:e]
            mb_mask = response_mask[s:e]
            mb_adv = advantages[s:e]
            mb_raw = raw_rewards[s:e]
            mb_old = old_log_probs_all[s:e] if old_log_probs_all is not None else None

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                policy_log_probs, token_entropy = _get_response_log_probs_and_entropy(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                )
                per_token_loss, meta = _compute_policy_gradient_loss(
                    policy_log_probs=policy_log_probs,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old,
                    cliprange=args.cliprange,
                )
                loss = _masked_mean(per_token_loss, mb_mask, dim=None) / float(
                    args.gradient_accumulation_steps
                )

            loss.backward()
            train_loss_accum += float(loss.item())
            entropy_accum += float(_masked_mean(token_entropy.float(), mb_mask, dim=None).item())

            if "clip_fraction" in meta:
                cf = _masked_mean(meta["clip_fraction"], mb_mask, dim=None)
                clip_frac_accum += float(cf.item())
                clip_frac_count += 1

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0).item()
        )
        optimizer.step()

        pbar.update(1)
        pbar.set_postfix(loss=f"{train_loss_accum:.4f}", reward=f"{reward_meta['reward_mean']:.3f}")

        train_metrics = {
            "train_step": step,
            "train/loss": train_loss_accum,
            "train/grad_norm": grad_norm,
            "train/token_entropy": entropy_accum / max(1, n_microbatches_per_rollout_batch),
            "train/reward_mean": reward_meta["reward_mean"],
            "train/reward_std": reward_meta["reward_std"],
            "train/reward_format_mean": reward_meta["reward_format_mean"],
            "train/reward_answer_mean": reward_meta["reward_answer_mean"],
        }
        if clip_frac_count:
            train_metrics["train/clip_fraction"] = clip_frac_accum / clip_frac_count
        if wb:
            wb.log(train_metrics)

        if step % args.eval_every == 0 or step == args.n_grpo_steps:
            load_policy_into_vllm_instance(policy, vllm_llm)
            val_reward = _eval_val_reward(
                vllm_llm=vllm_llm,
                prompt_template=prompt_template,
                val_questions=val_questions,
                val_ground_truths=val_gts,
                max_eval_examples=args.eval_n,
                max_tokens=args.eval_max_tokens,
            )
            eval_metrics = {"eval_step": step, "eval/val_reward": val_reward}
            print(json.dumps(eval_metrics, indent=2))
            if wb:
                wb.log(eval_metrics)

    pbar.close()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved GRPO checkpoint to {out_dir.resolve()}")
    if wb:
        wb.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--countdown-dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="validation")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=512)

    p.add_argument("--n-grpo-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--rollout-batch-size", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sampling-temperature", type=float, default=0.7)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument(
        "--loss-type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default="reinforce_with_baseline",
    )
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--use-std-normalization", action="store_true", default=True)
    p.add_argument("--no-std-normalization", action="store_false", dest="use_std_normalization")
    p.add_argument("--max-seq-length", type=int, default=4096)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:1")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--eval-max-tokens", type=int, default=1024)

    p.add_argument("--output-dir", type=Path, default=Path("outputs/grpo_run"))
    p.add_argument("--wandb-project", default="nyu-llm-reasoners-a3-grpo")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()

