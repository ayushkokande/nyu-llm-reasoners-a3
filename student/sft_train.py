"""Supervised fine-tuning for Qwen2.5-Math on Prime Intellect INTELLECT-MATH-SFT-Data.

Supports the full assignment SFT experiment:
  - Subset sizes {128, 256, 512, 1024, full}
  - wandb metrics with train_step / eval_step as x-axes
  - Gradient clipping 1.0
  - Periodic MATH accuracy evaluation
  - Optional vLLM on a second GPU for fast evaluation (2-GPU mode)

Single-GPU quick test (no vLLM, no wandb):

  uv run python -m student.sft_train --max-train-samples 128 --eval-every 10 \
    --math-eval-n 32 --max-steps 50 --no-wandb

Two-GPU cluster run (vLLM eval on cuda:1):

  uv run python -m student.sft_train --max-train-samples 512 \
    --vllm-device cuda:1 --learning-rate 2e-5 --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 --eval-every 50 --output-dir outputs/sft_n512

After training, evaluate the saved checkpoint with vLLM:

  uv run python -m student.evaluate --model outputs/sft_n512
"""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from student.drgrpo_grader import question_only_reward_fn


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    p = Path(__file__).parent / "prompts" / "intellect.prompt"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# wandb setup (matches handout Section 4.3)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenization helpers (matches run_tokenize_prompt_and_output in adapters.py)
# ---------------------------------------------------------------------------

def _encode_sft_example(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Encode one SFT example into shifted input_ids / labels / response_mask.

    Layout (matching ``run_tokenize_prompt_and_output``):
      full_ids  = prompt_ids ++ output_ids
      input_ids = full_ids[:-1]
      labels    = full_ids[1:]
      response_mask[i] = 1 iff labels[i] is a response token
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("expected last message role assistant")

    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    output = messages[-1]["content"]

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    output_ids = tokenizer.encode(output, add_special_tokens=False)

    if len(prompt_ids) + len(output_ids) > max_seq_length:
        budget = max_seq_length - len(prompt_ids)
        if budget < 1:
            prompt_ids = prompt_ids[: max_seq_length - 1]
            output_ids = output_ids[:1]
        else:
            output_ids = output_ids[:budget]

    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + output_ids
    full_len = len(full_ids)
    seq_len = full_len - 1

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
    labels = torch.tensor(full_ids[1:], dtype=torch.long)

    response_mask = torch.zeros(seq_len, dtype=torch.bool)
    start = prompt_len - 1
    end = full_len - 1
    if start < end and start >= 0:
        response_mask[start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def _collate_batch(
    batch: list[dict[str, Any]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(int(x["input_ids"].shape[0]) for x in batch)
    input_ids, labels, response_mask, attention_mask = [], [], [], []
    for x in batch:
        seq_len = int(x["input_ids"].shape[0])
        pad_n = max_len - seq_len
        ids = x["input_ids"].tolist()
        lab = x["labels"].tolist()
        msk = x["response_mask"].tolist()
        if pad_n:
            ids = ids + [pad_id] * pad_n
            lab = lab + [pad_id] * pad_n
            msk = msk + [False] * pad_n
        input_ids.append(ids)
        labels.append(lab)
        response_mask.append(msk)
        attention_mask.append([1] * seq_len + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_prime_intellect_dataset(
    tokenizer: Any,
    max_seq_length: int,
    max_train_samples: int | None,
    seed: int,
) -> Dataset:
    ds = load_dataset("PrimeIntellect/INTELLECT-MATH-SFT-Data", split="train")

    if max_train_samples is not None and max_train_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_train_samples))

    return ds.map(
        lambda ex: _encode_sft_example(ex["messages"], tokenizer, max_seq_length),
        remove_columns=ds.column_names,
    )


def _math_prompts_and_answers(
    split: str,
    n: int,
    seed: int,
    prompt_template: str,
) -> tuple[list[str], list[str]]:
    math_ds = load_dataset("hiyouga/math12k", split=split)
    rng = random.Random(seed)
    indices = list(range(len(math_ds)))
    rng.shuffle(indices)
    indices = indices[: min(n, len(indices))]
    prompts, gts = [], []
    for i in indices:
        ex = math_ds[int(i)]
        prompts.append(prompt_template + "\n\n" + ex["problem"])
        gts.append(ex["answer"])
    return prompts, gts


def _intellect_disk_prompts_and_answers(
    path: str,
    n: int,
) -> tuple[list[str], list[str]]:
    dataset = load_from_disk(path)
    if n:
        dataset = dataset.select(range(min(n, len(dataset))))
    prompts: list[str] = []
    gts: list[str] = []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))
    return prompts, gts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_accuracy_vllm(
    vllm_llm: Any,
    prompts: list[str],
    ground_truths: list[str],
    max_new_tokens: int,
) -> float:
    """Evaluate accuracy using vLLM (fast, batched)."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = vllm_llm.generate(prompts, params)
    correct = 0.0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        correct += question_only_reward_fn(text, ground_truths[i])["reward"]
    return correct / len(prompts) if prompts else 0.0


@torch.no_grad()
def _eval_accuracy_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    ground_truths: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> float:
    """Evaluate accuracy using transformers generate (slower, single-GPU fallback)."""
    model.eval()
    correct = 0.0
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    for prompt, gt in tqdm(
        list(zip(prompts, ground_truths)),
        desc="eval(generate)",
        leave=False,
    ):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        correct += question_only_reward_fn(text, gt)["reward"]
    model.train()
    return correct / len(prompts) if prompts else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_loop(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building training dataset...")
    train_ds = _build_prime_intellect_dataset(
        tokenizer,
        args.max_seq_length,
        args.max_train_samples,
        args.seed,
    )
    print(f"  {len(train_ds)} examples after tokenization")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)
    model.train()

    # --- optional vLLM on second GPU (handout 2-GPU mode) ---
    vllm_llm = None
    if args.vllm_device:
        from student.sft_vllm_utils import init_vllm
        print(f"Initializing vLLM on {args.vllm_device} ...")
        vllm_llm = init_vllm(
            model_id=args.model_id,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_mem,
        )

    dl = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_batch(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
        num_workers=args.num_workers,
        drop_last=False,
    )

    opt = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    micro_per_epoch = max(1, math.ceil(len(train_ds) / args.per_device_batch_size))
    opt_steps_per_epoch = max(1, math.ceil(micro_per_epoch / args.gradient_accumulation_steps))
    total_steps = args.num_epochs * opt_steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio else args.warmup_steps
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    wb = _wandb_init(args)
    train_step = 0
    eval_step = 0

    prompt_template = _load_prompt_template()
    math_prompts, math_gts = _math_prompts_and_answers(
        args.math_eval_split,
        args.math_eval_n,
        args.seed,
        prompt_template,
    )

    intellect_prompts: list[str] | None = None
    intellect_gts: list[str] | None = None
    if args.intellect_eval_path:
        try:
            intellect_prompts, intellect_gts = _intellect_disk_prompts_and_answers(
                args.intellect_eval_path, args.intellect_eval_n,
            )
        except Exception as e:
            print(f"Intellect eval skipped: {e}")

    print(f"Total optimizer steps: {total_steps} | warmup: {warmup_steps}")
    print(f"Micro-batches per epoch: {micro_per_epoch} | GA: {args.gradient_accumulation_steps}")

    use_amp = args.bf16 and device.type == "cuda"
    opt.zero_grad(set_to_none=True)
    accum = 0
    running_nll = 0.0
    running_ntok = 0

    pbar = tqdm(total=total_steps, desc="train")
    it = cycle(dl)

    while train_step < total_steps:
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

        # Adapter-style loss: -sum(log_probs * mask) / (2 * GA)
        log_probs_all = torch.log_softmax(logits.float(), dim=-1)
        log_probs = torch.gather(
            log_probs_all, dim=-1, index=batch["labels"].unsqueeze(-1)
        ).squeeze(-1)

        mask_f = batch["response_mask"].to(log_probs.dtype)
        masked = log_probs * mask_f
        loss = -masked.sum() / (2.0 * args.gradient_accumulation_steps)
        loss.backward()

        n_resp_tokens = int(batch["response_mask"].sum().item())
        running_nll += float(-masked.sum().item())
        running_ntok += n_resp_tokens
        accum += 1

        if accum < args.gradient_accumulation_steps:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        accum = 0
        train_step += 1

        nll_per_token = running_nll / max(running_ntok, 1)
        pbar.update(1)
        pbar.set_postfix(nll=f"{nll_per_token:.4f}")
        if wb:
            wb.log(
                {"train/loss": nll_per_token, "train/lr": sched.get_last_lr()[0]},
                step=train_step,
            )
        running_nll = 0.0
        running_ntok = 0

        # --- periodic evaluation ---
        if train_step % args.eval_every == 0 or train_step >= total_steps:
            eval_step += 1
            metrics: dict[str, Any] = {}

            if vllm_llm is not None:
                from student.sft_vllm_utils import load_policy_into_vllm_instance
                load_policy_into_vllm_instance(model, vllm_llm)
                math_acc = _eval_accuracy_vllm(vllm_llm, math_prompts, math_gts, args.eval_max_new_tokens)
            else:
                math_acc = _eval_accuracy_generate(
                    model, tokenizer, math_prompts, math_gts, device, args.eval_max_new_tokens,
                )

            metrics["eval/math_accuracy"] = math_acc

            if intellect_prompts and intellect_gts:
                if vllm_llm is not None:
                    metrics["eval/intellect_accuracy"] = _eval_accuracy_vllm(
                        vllm_llm, intellect_prompts, intellect_gts, args.eval_max_new_tokens,
                    )
                else:
                    metrics["eval/intellect_accuracy"] = _eval_accuracy_generate(
                        model, tokenizer, intellect_prompts, intellect_gts,
                        device, args.eval_max_new_tokens,
                    )

            print(json.dumps({"eval_step": eval_step, **metrics}, indent=2))
            if wb:
                wb.log(metrics, step=eval_step)

    pbar.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved checkpoint to {out_dir.resolve()}")
    if wb:
        wb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--max-train-samples", type=int, default=None,
                    help="128, 256, 512, 1024, or omit for full dataset")
    p.add_argument("--max-seq-length", type=int, default=4096)

    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--eval-every", type=int, default=200,
                    help="Evaluate every N optimizer steps")
    p.add_argument("--math-eval-split", default="train",
                    help="hiyouga/math12k split for periodic accuracy")
    p.add_argument("--math-eval-n", type=int, default=128)
    p.add_argument("--intellect-eval-path", default=None,
                    help="Path to load_from_disk test set for Intellect eval")
    p.add_argument("--intellect-eval-n", type=int, default=128)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)

    p.add_argument("--vllm-device", default=None,
                    help="Device for vLLM eval, e.g. cuda:1. Omit to use transformers.generate.")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)

    p.add_argument("--output-dir", type=Path, default=Path("outputs/sft_run"))
    p.add_argument("--wandb-project", default="nyu-llm-reasoners-a3-sft")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be >= 1")
    train_loop(args)


if __name__ == "__main__":
    main()
