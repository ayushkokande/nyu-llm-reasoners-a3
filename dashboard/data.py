"""Result numbers for the dashboard — single source of truth for the UI.

All values are the actual measured results from the experiments.
"""

MODEL = "Qwen/Qwen2.5-Math-1.5B"

# ---------------------------------------------------------------- Stage 0: baseline
BASELINE = {
    "math_accuracy": 0.6080,          # 304 / 500
    "n_total": 500,
    "categories": [
        # (label, count, pct, meaning)
        ("Correct (fmt=1, ans=1)", 304, 60.80, "Parseable AND mathematically correct"),
        ("Wrong  (fmt=1, ans=0)", 166, 33.20, "Parseable boxed answer, but incorrect value"),
        ("No box (fmt=0, ans=0)", 30, 6.00, "No \\boxed{} signal — generation breakdown"),
    ],
    # representative failure indices pulled from the report
    "fmt0_examples": [9, 48, 50, 64, 94, 100, 103, 108, 110, 277],
    "ans0_examples": [7, 11, 14, 15, 17, 18, 25, 32, 71, 120],
    "takeaway": (
        "Bottleneck is the **model**, not the parser. 33.2% of outputs are "
        "parseable-but-wrong (parser working as intended); the 6% format failures "
        "are truncation, repetition loops, or prose-only answers — all generation "
        "failures, not parsing bugs."
    ),
}

# ---------------------------------------------------------------- Stage 1: SFT
# Table 2 (peak val acc) + Table 3 (final NLL) from the report.
SFT_SWEEP = [
    # (dataset_size, total_steps, final_nll, peak_val_acc_pct)
    ("128",   256,   0.1710, 78.15),
    ("256",   512,   0.1410, 76.56),
    ("512",   1024,  0.1838, 74.22),
    ("1024",  2048,  0.1028, 73.44),
    ("Full",  1600,  0.0650, 73.44),
]
SFT_BEST = {
    "run": "128-example",
    "selection_rule": "highest MATH validation accuracy",
    "math_test": 0.657,
    "intellect_test": 0.634,
    "hparams": "lr=5e-5 · batch=1 · grad-accum=2 (eff batch 2) · 4 epochs",
    "takeaway": (
        "Best checkpoint came from the **smallest** (128-example) run. Simply adding "
        "SFT data did not improve downstream test accuracy under a fixed compute "
        "budget — bigger sets need more steps / retuned hyperparameters to pay off."
    ),
}

# ---------------------------------------------------------------- Stage 2: GRPO
GRPO_MAIN = {
    "task": "Countdown (Jiayi-Pan/Countdown-Tasks-3to4)",
    "start_reward": 0.153,
    "peak_reward": 0.7646,
    "rollout_steps": 500,
    "takeaway": (
        "GRPO lifted Countdown dev accuracy from **0.153 → 0.7646** over 500 rollout "
        "steps — a decisively upward trajectory, confirming the policy was refined by "
        "reward rather than drifting on noise."
    ),
}

# Each ablation: rows of (setting, best_dev, final_dev, test_acc_or_None) + figure + notes
GRPO_ABLATIONS = {
    "Learning rate sweep": {
        "fig": "fig3_lr_sweep.png",
        "rows": [
            ("lr = 1e-5  (optimal)", 0.48, 0.48, None),
            ("lr = 3e-5  (collapsed @160)", None, None, None),
        ],
        "cols": ["Setting", "≈ Final Dev Acc", "—", "—"],
        "notes": (
            "1e-5 is the sweet spot: high enough to learn efficiently, low enough to "
            "stay stable (final ≈48%, clears the 30% bar). 3e-5 made rapid early gains "
            "then drifted too far and **collapsed after step 160**."
        ),
    },
    "Baseline (loss type)": {
        "fig": "fig4_loss_type.png",
        "rows": [
            ("no_baseline", None, None, None),
            ("reinforce_with_baseline", None, None, None),
        ],
        "cols": ["Setting", "—", "—", "—"],
        "notes": (
            "Curves converge to similar final accuracy. `no_baseline` led early with a "
            "slight peak edge; `reinforce_with_baseline` was steadier with lower gradient "
            "variance. The baseline acts as **variance reduction**, not an outcome changer "
            "for this on-policy budget."
        ),
    },
    "Length normalization": {
        "fig": "fig5_length_norm.png",
        "rows": [
            ("masked_mean", 0.5282, 0.5104, 0.4386),
            ("masked_normalize", 0.4484, 0.4351, 0.3812),
        ],
        "cols": ["Setting", "Best Dev", "Final Dev", "Test Acc"],
        "notes": (
            "`masked_mean` wins (test 0.4386 vs 0.3812). Dividing by the unmasked-token "
            "count keeps each token's gradient proportional to its advantage; "
            "`masked_normalize`'s fixed-constant denominator erases advantage magnitude "
            "and yields noisier gradient norms that hit the clip threshold more often."
        ),
    },
    "Std normalization": {
        "fig": "fig6_std_norm.png",
        "rows": [
            ("use_std = True", 0.5012, 0.4571, 0.4147),
            ("use_std = False (Dr. GRPO)", 0.4052, 0.3518, 0.3426),
        ],
        "cols": ["Setting", "Best Dev", "Final Dev", "Test Acc"],
        "notes": (
            "`use_std=True` wins here (final 0.4571 vs 0.3518). Removing std gives a strong "
            "early signal, but as more question groups become uniformly right/wrong the "
            "advantage collapses to ~0 and the curve plateaus. Std normalization keeps the "
            "signal alive late in training."
        ),
    },
}

# Two rollout examples shown in the report (both scored 0 reward — format/answer miss)
GRPO_ROLLOUTS = [
    {
        "step": 1,
        "reward": "{reward: 0.0, format: 0.0, answer: 0.0}",
        "text": (
            "Step 1: subtract 29 from 71 -> 42.\n"
            "Step 2: 42 - 39 = 3.\n"
            "Step 3: multiply 3 by 21 (61 minus 40) -> 63.4\n"
            "Step 4: subtract 2 from 63 -> 61.\n"
            "<answer>\n(71 - 29 - 39) * 21 - 2"
        ),
    },
    {
        "step": 500,
        "reward": "{reward: 0.0, format: 0.0, answer: 0.0}",
        "text": (
            "Eliminate 29 (largest, unhelpful) -> [71, 39, 2].\n"
            "Try 71 / 2 = 35.5 ... add 39 -> 74.5 (too high).\n"
            "71 - 39 = 32, close to 52. Add 2 -> 34 (too low).\n"
            "<answer>\nStep 1: 71 - 39 = 32\nStep 2: 32 + 20 = 52"
        ),
    },
]
GRPO_ROLLOUT_NOTE = (
    "Both traces are fluent but score **zero** — they miss the strict "
    "`</think> <answer>…</answer>` format or fail final answer extraction. Fluent "
    "reasoning alone earns no reward; this is exactly the signal sparsity that can "
    "stall GRPO early."
)

# Preset examples for the live Reward Playground (response, target, reward_type, blurb)
PLAYGROUND_PRESETS = {
    "Correct — fraction vs decimal": (
        r"After simplifying, the result is \boxed{1/2}.", "0.5", "Answer match",
        "The model writes 1/2, the target is 0.5 — the reward checks mathematical "
        "equivalence, so this earns full reward.",
    ),
    "Correct — LaTeX \\frac": (
        r"Therefore the probability is \boxed{\frac{1}{2}}.", "0.5", "Answer match",
        "LaTeX \\frac{1}{2} normalizes to 0.5 → full reward.",
    ),
    "Wrong answer": (
        r"The angle measures \boxed{100.30} degrees.", "90", "Answer match",
        "A clean, parseable answer — but the value is wrong, so no reward.",
    ),
    "Answer only in prose": (
        "The final answer is 1.25 — it should be one and a quarter.", "1.25", "Answer match",
        "Correct in prose, but the answer isn't in the expected form, so nothing is "
        "extracted — no reward.",
    ),
    "Format-gated — tags present": (
        r"work here </think> <answer>\boxed{0.5}</answer>", "1/2", "Format-gated",
        "The format-gated reward requires the answer wrapped in the expected tags. "
        "Tags present and correct → full reward.",
    ),
    "Format-gated — tags missing": (
        r"The answer is \boxed{0.5}", "0.5", "Format-gated",
        "Right answer, but the required tags are missing — the format-gated reward "
        "gives zero.",
    ),
}
