"""Minimal evaluation script for MATH and Intellect test sets.

Pipeline per run: load a HuggingFace model into vLLM -> wrap each test
question in a prompt template -> greedily generate a response -> grade the
response against the ground-truth answer with `question_only_reward_fn` ->
report accuracy (mean reward).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

# `datasets` pulls MATH from the HF Hub; `load_from_disk` reads the local
# Intellect split. tqdm = progress bars during grading.
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# The verified-reward grader: (response, ground_truth) -> dict with keys
# "reward", "format_reward", "answer_reward". This is the eval metric.
from .drgrpo_grader import question_only_reward_fn

# vLLM is heavy and CUDA-only, so it is imported lazily inside functions
# (see below). This TYPE_CHECKING block only feeds type hints, never runs.
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def load_prompt(name: str = "intellect") -> str:
    # Read a prompt template from reasoning_rl/prompts/<name>.prompt. The question
    # is appended to this string later to form the full model input.
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(
    llm: LLM,
    prompts: list[str],
    ground_truths: list[str],
) -> float:
    """Generate one response per prompt and return mean reward (accuracy)."""
    from vllm import SamplingParams

    # temperature=0.0 gives deterministic greedy decoding; cap each generation
    # at 2048 tokens.
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    # Batched generation: vLLM returns outputs in the same order as `prompts`.
    outputs = llm.generate(prompts, params)

    # Grade each generation against its ground truth; sum the binary rewards.
    correct = 0.0
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text  # first (only) sampled completion
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]

    # Accuracy = fraction correct. Guard against empty input.
    return correct / len(outputs) if outputs else 0.0


def evaluate_math_baseline(
    llm: LLM,
    prompt_template: str,
    max_examples: int | None,
    log_fp: TextIO,
) -> tuple[float, dict[str, int]]:
    """Run the zero-shot MATH baseline and bucket model outputs.

    Same generate-and-grade loop as `evaluate`, but additionally (a) buckets
    every generation by format/answer reward into three diagnostic categories
    and (b) logs every example to a JSONL file for manual inspection.
    Returns (accuracy, bucket counts).
    """
    from vllm import SamplingParams

    # Load the MATH-12k test split, then trim to at most `max_examples`.
    math_ds = load_dataset("hiyouga/math12k", split="test")
    n_all = len(math_ds)
    n = n_all if max_examples is None else min(max_examples, n_all)
    math_ds = math_ds.select(range(n))

    # Build full prompts (template + question) and keep parallel lists of the
    # gold answers and raw problem text for logging.
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]
    problems = [ex["problem"] for ex in math_ds]

    # Greedy decode all prompts in one batched call.
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    # Three reward buckets:
    # (1) format=1 answer=1  (2) format=1 answer=0  (3) format=0 (answer=0)
    c1 = c2 = c3 = 0
    total_reward = 0.0

    for i, output in enumerate(tqdm(outputs, desc="MATH baseline")):
        text = output.outputs[0].text
        gt = gts[i]
        # Grade: format_reward = was the \boxed{} format produced;
        # answer_reward = was the boxed value correct.
        r = question_only_reward_fn(text, gt)
        fr, ar = int(r["format_reward"]), int(r["answer_reward"])
        total_reward += r["reward"]

        # Tally into the bucket this generation falls in.
        if fr == 1 and ar == 1:
            c1 += 1
        elif fr == 1 and ar == 0:
            c2 += 1
        else:
            c3 += 1

        # Append one JSON record per example; flush so a crash keeps partial
        # results. ensure_ascii=False preserves LaTeX/unicode verbatim.
        log_fp.write(
            json.dumps(
                {
                    "index": i,
                    "problem": problems[i],
                    "ground_truth": gt,
                    "model_output": text,
                    "format_reward": r["format_reward"],
                    "answer_reward": r["answer_reward"],
                    "reward": r["reward"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_fp.flush()

    # Accuracy = mean reward over the n graded examples.
    acc = total_reward / n if n else 0.0
    return acc, {"cat_both_1": c1, "cat_fmt1_ans0": c2, "cat_fmt0": c3, "n": n}


def main() -> None:
    # --- CLI args ---------------------------------------------------------
    # --model: HF model id/path. --max-examples: cap per dataset (default 500
    # = the MATH test set size). --gpu-memory-utilization: fraction of VRAM
    # vLLM may grab. --math-only: skip the Intellect split. --math-baseline:
    # run the bucketed MATH baseline and write a JSONL log.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math_train_dev_test/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--math-only", action="store_true", help="skip Intellect")
    parser.add_argument(
        "--math-baseline",
        action="store_true",
        help="MATH only: jsonl log + reward bucket counts",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None, help="required with --math-baseline")
    args = parser.parse_args()

    # Baseline mode must have somewhere to write the per-example log.
    if args.math_baseline and args.log_jsonl is None:
        print("--math-baseline needs --log-jsonl", file=sys.stderr)
        sys.exit(1)

    # All paths share the same Intellect prompt template.
    prompt_template = load_prompt("intellect")

    # Spin up the inference engine (downloads the model on first run).
    # trust_remote_code allows Qwen's custom modeling files to load.
    from vllm import LLM

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # --- Mode 1: bucketed MATH baseline ----------------------------------
    # Runs the baseline, prints the 3 category counts + accuracy, then exits.
    if args.math_baseline:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_jsonl, "w", encoding="utf-8") as log_fp:
            acc, counts = evaluate_math_baseline(
                llm, prompt_template, args.max_examples, log_fp
            )
        n = counts["n"]
        print("\n=== math_baseline ===\n")
        print(f"(1) format=1, answer=1:     {counts['cat_both_1']:5d}  ({100 * counts['cat_both_1'] / n:.2f}%)")
        print(f"(2) format=1, answer=0:     {counts['cat_fmt1_ans0']:5d}  ({100 * counts['cat_fmt1_ans0'] / n:.2f}%)")
        print(f"(3) format=0, answer=0:     {counts['cat_fmt0']:5d}  ({100 * counts['cat_fmt0'] / n:.2f}%)")
        print(f"MATH acc: {acc:.4f}")
        print(args.log_jsonl.resolve())
        return

    # --- Mode 2 (default): plain accuracy on Intellect + MATH -------------
    # Intellect split (unless --math-only). Loaded from local disk; skipped
    # gracefully if the path is missing.
    if not args.math_only:
        print(f"\n=== Intellect Test ({args.intellect_path}) ===")
        try:
            dataset = load_from_disk(args.intellect_path)
        except Exception as e:
            print(f"intellect skip: {e}", file=sys.stderr)
        else:
            if args.max_examples:
                dataset = dataset.select(range(min(args.max_examples, len(dataset))))
            # Intellect examples are chat-formatted: flatten the system+user
            # messages into a single prompt string, and pull the gold answer.
            prompts, gts = [], []
            for ex in dataset:
                msgs = ex.get("messages", [])
                sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
                gts.append(ex.get("ground_truth", ""))
            print(f"[Sample] {prompts[0][:200]}...")
            acc = evaluate(llm, prompts, gts)
            print(f"Intellect Accuracy: {acc:.4f}")

    # MATH split: build prompts from the template, generate, grade, report.
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]
    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts)
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
