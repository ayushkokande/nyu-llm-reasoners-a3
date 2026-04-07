"""Minimal evaluation script for MATH and Intellect test sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(
    llm: LLM,
    prompts: list[str],
    ground_truths: list[str],
) -> float:
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0.0
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]

    return correct / len(outputs) if outputs else 0.0


def evaluate_math_baseline(
    llm: LLM,
    prompt_template: str,
    max_examples: int | None,
    log_fp: TextIO,
) -> tuple[float, dict[str, int]]:
    math_ds = load_dataset("hiyouga/math12k", split="test")
    n_all = len(math_ds)
    n = n_all if max_examples is None else min(max_examples, n_all)
    math_ds = math_ds.select(range(n))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]
    problems = [ex["problem"] for ex in math_ds]

    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    # (1) format=1 answer=1  (2) format=1 answer=0  (3) format=0 (answer=0)
    c1 = c2 = c3 = 0
    total_reward = 0.0

    for i, output in enumerate(tqdm(outputs, desc="MATH baseline")):
        text = output.outputs[0].text
        gt = gts[i]
        r = question_only_reward_fn(text, gt)
        fr, ar = int(r["format_reward"]), int(r["answer_reward"])
        total_reward += r["reward"]

        if fr == 1 and ar == 1:
            c1 += 1
        elif fr == 1 and ar == 0:
            c2 += 1
        else:
            c3 += 1

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

    acc = total_reward / n if n else 0.0
    return acc, {"cat_both_1": c1, "cat_fmt1_ans0": c2, "cat_fmt0": c3, "n": n}


def main() -> None:
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

    if args.math_baseline and args.log_jsonl is None:
        print("--math-baseline needs --log-jsonl", file=sys.stderr)
        sys.exit(1)

    prompt_template = load_prompt("intellect")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

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

    if not args.math_only:
        print(f"\n=== Intellect Test ({args.intellect_path}) ===")
        try:
            dataset = load_from_disk(args.intellect_path)
        except Exception as e:
            print(f"intellect skip: {e}", file=sys.stderr)
        else:
            if args.max_examples:
                dataset = dataset.select(range(min(args.max_examples, len(dataset))))
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
