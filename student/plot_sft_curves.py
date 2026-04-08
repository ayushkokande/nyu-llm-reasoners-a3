"""Plot SFT training loss and MATH validation accuracy from wandb or CSV.

Usage (from wandb):

  uv run python -m student.plot_sft_curves \
    --entity YOUR_WANDB_ENTITY \
    --project nyu-llm-reasoners-a3-sft \
    --filter-prefix sft_n512 \
    --out-dir plots

Usage (from exported CSVs):

  uv run python -m student.plot_sft_curves \
    --csv-dir wandb_exports/ \
    --out-dir plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _fetch_wandb_runs(entity: str, project: str, prefix: str | None) -> list[dict[str, Any]]:
    """Fetch run histories from wandb API."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}"
    runs = api.runs(path)

    results = []
    for run in runs:
        if prefix and not run.name.startswith(prefix):
            continue
        history = run.history(pandas=False)
        train_steps, train_losses, eval_steps, eval_accs = [], [], [], []
        for row in history:
            if "train/loss" in row and row["train/loss"] is not None:
                step = row.get("train_step", row.get("_step", 0))
                train_steps.append(step)
                train_losses.append(row["train/loss"])
            if "eval/math_accuracy" in row and row["eval/math_accuracy"] is not None:
                step = row.get("eval_step", row.get("_step", 0))
                eval_steps.append(step)
                eval_accs.append(row["eval/math_accuracy"])
        results.append({
            "name": run.name,
            "config": dict(run.config),
            "train_steps": train_steps,
            "train_losses": train_losses,
            "eval_steps": eval_steps,
            "eval_accs": eval_accs,
        })
    return results


def _load_csv_runs(csv_dir: str) -> list[dict[str, Any]]:
    """Load runs from CSV files exported from wandb."""
    import csv

    results = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        name = fname.replace(".csv", "")
        train_steps, train_losses, eval_steps, eval_accs = [], [], [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "train/loss" in row and row["train/loss"]:
                    step = int(row.get("train_step", row.get("_step", "0")))
                    train_steps.append(step)
                    train_losses.append(float(row["train/loss"]))
                if "eval/math_accuracy" in row and row["eval/math_accuracy"]:
                    step = int(row.get("eval_step", row.get("_step", "0")))
                    eval_steps.append(step)
                    eval_accs.append(float(row["eval/math_accuracy"]))
        results.append({
            "name": name,
            "config": {},
            "train_steps": train_steps,
            "train_losses": train_losses,
            "eval_steps": eval_steps,
            "eval_accs": eval_accs,
        })
    return results


def plot_curves(runs: list[dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))

    for run in runs:
        label = run["name"]
        if run["train_steps"]:
            ax_loss.plot(run["train_steps"], run["train_losses"], label=label, alpha=0.8)
        if run["eval_steps"]:
            ax_acc.plot(run["eval_steps"], run["eval_accs"], label=label, marker="o", alpha=0.8)

    ax_loss.set_xlabel("train_step")
    ax_loss.set_ylabel("train/loss (NLL per token)")
    ax_loss.set_title("SFT Training Loss")
    ax_loss.legend(fontsize=7, loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(out_dir, "sft_train_loss.png"), dpi=150)
    print(f"Saved {out_dir}/sft_train_loss.png")

    ax_acc.set_xlabel("eval_step")
    ax_acc.set_ylabel("eval/math_accuracy")
    ax_acc.set_title("SFT MATH Validation Accuracy")
    ax_acc.legend(fontsize=7, loc="lower right")
    ax_acc.grid(True, alpha=0.3)
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join(out_dir, "sft_math_accuracy.png"), dpi=150)
    print(f"Saved {out_dir}/sft_math_accuracy.png")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None, help="wandb entity (username or team)")
    parser.add_argument("--project", default="nyu-llm-reasoners-a3-sft")
    parser.add_argument("--filter-prefix", default=None, help="Only plot runs whose name starts with this")
    parser.add_argument("--csv-dir", default=None, help="Load from exported CSVs instead of wandb")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    if args.csv_dir:
        runs = _load_csv_runs(args.csv_dir)
    else:
        if not args.entity:
            raise SystemExit("--entity required when fetching from wandb")
        runs = _fetch_wandb_runs(args.entity, args.project, args.filter_prefix)

    if not runs:
        raise SystemExit("No runs found")
    print(f"Found {len(runs)} run(s)")
    plot_curves(runs, args.out_dir)


if __name__ == "__main__":
    main()
