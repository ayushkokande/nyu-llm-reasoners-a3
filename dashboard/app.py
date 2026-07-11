"""Interactive showcase: teaching a 1.5B language model to reason about math
with supervised finetuning (SFT) and Group-Relative Policy Optimization (GRPO).

Run:
    uv run --with streamlit streamlit run dashboard/app.py

The Reward Playground tab runs the math reward function live in-browser — it is
pure Python, so it works on CPU with no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

import data as D

REPO = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The reward function is pure-Python -> safe to import and run on CPU.
try:
    from reasoning_rl.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

    REWARDS = {"Answer match": question_only_reward_fn, "Format-gated": r1_zero_reward_fn}
    REWARD_OK = True
    REWARD_ERR = ""
except Exception as e:  # pragma: no cover - defensive
    REWARDS, REWARD_OK, REWARD_ERR = {}, False, repr(e)


st.set_page_config(page_title="Math Reasoning RL", page_icon="🧠", layout="wide")


def img(name: str, caption: str = "") -> None:
    """Show a result figure if present."""
    p = ASSETS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"figure not found: {name}")


# ----------------------------------------------------------------- header
st.title("🧠 Math Reasoning with Reinforcement Learning")
st.caption(
    f"Improving **{D.MODEL}**'s mathematical reasoning through supervised "
    "finetuning (SFT) and Group-Relative Policy Optimization (GRPO)."
)

h0, h1, h2 = st.columns(3)
h0.metric("Zero-shot baseline (MATH)", f"{D.BASELINE['math_accuracy']*100:.1f}%",
          help="Base model, greedy decoding, 500-example MATH test set")
h1.metric("Best SFT validation peak", f"{D.SFT_SWEEP[0][3]:.2f}%".rstrip("0").rstrip("."),
          help="Best supervised-finetuning run, MATH validation accuracy")
h2.metric("GRPO reasoning reward", f"{D.GRPO_MAIN['peak_reward']:.3f}",
          delta=f"+{D.GRPO_MAIN['peak_reward'] - D.GRPO_MAIN['start_reward']:.3f} from start",
          help="Peak Countdown validation reward over 500 RL steps")

tabs = st.tabs(
    ["📋 Overview", "🎮 Reward Playground", "Baseline", "SFT", "GRPO", "📐 Approach"]
)

# ----------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("The pipeline")
    st.markdown(
        """
```
   Qwen2.5-Math-1.5B (base)
            │
   ┌────────▼─────────┐   1 · EVALUATE
   │  zero-shot eval  │   greedy decode → extract final answer → check correctness
   └────────┬─────────┘   60.8% on MATH (the reference everything is compared against)
            │
   ┌────────▼─────────┐   2 · SFT  (imitation)
   │ supervised       │   token-level cross-entropy on the response tokens only
   │ finetune         │   sweep 128…full examples; best = 128-example run
   └────────┬─────────┘
            │
   ┌────────▼─────────┐   3 · GRPO  (reinforcement)
   │ group-relative   │   sample G answers/prompt, advantage = reward − group mean
   │ policy optim.    │   Countdown reward 0.153 → 0.765
   └──────────────────┘
```
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What each stage does")
        st.markdown(
            "- **Baseline** — measure the untrained model and analyze *why* it fails "
            "(reasoning mistake vs. formatting/extraction miss).\n"
            "- **SFT** — warm-start the policy by imitating worked reasoning traces; "
            "study how accuracy scales with dataset size.\n"
            "- **GRPO** — refine the policy with *reward* (no value network) and ablate "
            "the loss design: learning rate, baseline, length norm, std norm."
        )
    with c2:
        st.markdown("#### Headline results")
        st.markdown(
            f"- MATH zero-shot baseline: **{D.BASELINE['math_accuracy']*100:.1f}%**\n"
            f"- Best SFT model: MATH test **{D.SFT_BEST['math_test']:.3f}**, "
            f"in-domain test **{D.SFT_BEST['intellect_test']:.3f}**\n"
            f"- GRPO on Countdown: **{D.GRPO_MAIN['start_reward']:.3f} → "
            f"{D.GRPO_MAIN['peak_reward']:.3f}** over {D.GRPO_MAIN['rollout_steps']} steps\n"
            f"- Best RL settings: **masked_mean** + **std-norm on** + **lr 1e-5**"
        )
    st.info("👉 The **Reward Playground** tab runs the math reward function live — no GPU needed.")

# ----------------------------------------------------------------- Playground
with tabs[1]:
    st.subheader("🎮 Reward Playground")
    st.caption(
        "A rule-based reward for math answers, running live on CPU. It pulls the "
        "model's final answer out of its response and checks **mathematical "
        "equivalence** to the target (handling fractions, decimals, and LaTeX). "
        "Type an answer and see the reward breakdown."
    )

    if not REWARD_OK:
        st.error(f"Reward function unavailable.\n\n`{REWARD_ERR}`")
    else:
        # presets populate the widgets via session_state before they render
        st.markdown("**Try an example:**")
        pcols = st.columns(3)
        preset_items = list(D.PLAYGROUND_PRESETS.items())
        for i, (label, (resp, gt, fn, _blurb)) in enumerate(preset_items):
            def _set(resp=resp, gt=gt, fn=fn, blurb=_blurb):
                st.session_state["pg_resp"] = resp
                st.session_state["pg_gt"] = gt
                st.session_state["pg_fn"] = fn
                st.session_state["pg_blurb"] = blurb
            pcols[i % 3].button(label, on_click=_set, width="stretch")

        _first = D.PLAYGROUND_PRESETS["Correct — fraction vs decimal"]
        st.session_state.setdefault("pg_resp", _first[0])
        st.session_state.setdefault("pg_gt", _first[1])
        st.session_state.setdefault("pg_fn", _first[2])
        st.session_state.setdefault("pg_blurb", _first[3])

        left, right = st.columns([3, 2])
        with left:
            fn_name = st.radio(
                "Reward type", list(REWARDS), horizontal=True, key="pg_fn",
                help="Answer match: checks the final answer. "
                     "Format-gated: also requires the answer wrapped in the expected tags.",
            )
            response = st.text_area("Model response", key="pg_resp", height=180)
            ground_truth = st.text_input("Target answer", key="pg_gt")
            go = st.button("⚖️  Compute reward", type="primary")
        with right:
            if st.session_state.get("pg_blurb"):
                st.caption("What this shows:")
                st.markdown(f"> {st.session_state['pg_blurb']}")

        if go or response:
            try:
                out = REWARDS[fn_name](response, ground_truth)
                m1, m2, m3 = st.columns(3)
                m1.metric("format", f"{out['format_reward']:.0f}")
                m2.metric("answer", f"{out['answer_reward']:.0f}")
                m3.metric("reward", f"{out['reward']:.0f}")
                if out["reward"] >= 1.0:
                    st.success("✅ Correct and well-formatted — full reward.")
                elif out["format_reward"] >= 1.0:
                    st.warning("⚠️ Parseable but the value is wrong — no reward.")
                else:
                    st.error("❌ No valid answer could be extracted from the response.")
                st.caption("Format alone earns no reward — this prevents the policy from "
                           "reward-hacking the output format without actually solving the problem.")
                st.code(repr(out), language="python")
            except Exception as e:
                st.exception(e)

# ----------------------------------------------------------------- Baseline
with tabs[2]:
    st.subheader("Baseline — zero-shot MATH accuracy")
    st.metric("Accuracy (MATH test, 500 problems)", f"{D.BASELINE['math_accuracy']*100:.2f}%",
              help="304 / 500 correct")

    rows = [{"Outcome": c[0], "Count": c[1], "%": c[2], "Meaning": c[3]} for c in D.BASELINE["categories"]]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.bar_chart(
        {"count": {c[0]: c[1] for c in D.BASELINE["categories"]}},
        horizontal=True,
    )

    st.markdown(f"**Takeaway.** {D.BASELINE['takeaway']}")
    e1, e2 = st.columns(2)
    e1.markdown("**No answer extracted** (generation failure):\n\n" +
                ", ".join(f"`#{i}`" for i in D.BASELINE["fmt0_examples"]))
    e2.markdown("**Wrong answer** (parseable, incorrect):\n\n" +
                ", ".join(f"`#{i}`" for i in D.BASELINE["ans0_examples"]))

# ----------------------------------------------------------------- SFT
with tabs[3]:
    st.subheader("SFT — supervised finetuning (dataset-size study)")
    img("fig1_sft_val_acc.png", "Validation MATH accuracy vs optimizer steps, per training-set size")

    rows = [
        {"Dataset size": s[0], "Total steps": s[1], "Final loss (NLL)": s[2], "Peak val acc (%)": s[3]}
        for s in D.SFT_SWEEP
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    b = D.SFT_BEST
    st.markdown(f"#### Best model — {b['run']} run")
    bc1, bc2 = st.columns(2)
    bc1.metric("MATH test accuracy", f"{b['math_test']:.3f}")
    bc2.metric("In-domain test accuracy", f"{b['intellect_test']:.3f}")
    st.caption(f"Selected by {b['selection_rule']} · {b['hparams']}")
    st.markdown(f"**Takeaway.** {b['takeaway']}")

# ----------------------------------------------------------------- GRPO
with tabs[4]:
    st.subheader("GRPO — reinforcement learning on Countdown")
    g = D.GRPO_MAIN
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Start reward", f"{g['start_reward']:.3f}")
    gc2.metric("Peak reward", f"{g['peak_reward']:.3f}", delta=f"+{g['peak_reward']-g['start_reward']:.3f}")
    gc3.metric("RL steps", g["rollout_steps"])
    img("fig2_grpo_reward.png", "Countdown validation reward vs RL step")
    st.markdown(f"**Takeaway.** {g['takeaway']}")

    st.divider()
    st.markdown("### Ablations")
    for name, ab in D.GRPO_ABLATIONS.items():
        with st.expander(name, expanded=(name == "Learning rate sweep")):
            img(ab["fig"])

            def _cell(v):
                # homogeneous strings per column -> avoids Arrow mixed-type errors
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.4f}".rstrip("0").rstrip(".")
                return str(v)

            table = []
            for r in ab["rows"]:
                row = {ab["cols"][0]: r[0]}
                for ci, val in zip(ab["cols"][1:], r[1:]):
                    if ci != "—":
                        row[ci] = _cell(val)
                table.append(row)
            st.dataframe(table, width="stretch", hide_index=True)
            st.markdown(ab["notes"])

    st.divider()
    st.markdown("### Example model responses")
    st.caption(D.GRPO_ROLLOUT_NOTE)
    for r in D.GRPO_ROLLOUTS:
        st.markdown(f"**RL step {r['step']}** — `{r['reward']}`")
        st.code(r["text"])

# ----------------------------------------------------------------- Approach
with tabs[5]:
    st.subheader("📐 Approach & methods")
    st.markdown(
        """
Three stages turn a base language model into a stronger math reasoner.

#### 1 · Evaluation
Greedy-decode the base model on held-out math problems, extract its final answer,
and check mathematical equivalence to the target. This zero-shot number is the
reference every later stage is measured against, and the failure analysis
separates *reasoning* mistakes from *formatting / extraction* misses.

#### 2 · Supervised finetuning (SFT)
Imitation learning. Each example pairs a math prompt with a worked
chain-of-thought and final answer; the loss is **token-level cross-entropy
computed only on the response tokens** (a response mask zeroes out prompt and
padding). This warm-starts a competent policy before any reward signal. I swept
the number of unique training examples to study data-efficiency.

#### 3 · Group-Relative Policy Optimization (GRPO)
Reinforcement learning that maximizes task reward **without a learned value
network**. For each prompt it samples a *group* of G responses and uses the
group's mean reward as the baseline, so each response's advantage is
`reward − group mean` (optionally divided by the group std). Above-average
responses are reinforced, below-average ones suppressed.

#### Reward design
A rule-based reward extracts the model's final answer and checks mathematical
equivalence to the target (fractions reduced, decimals and LaTeX normalized,
symbolic equivalence as a fallback). Crucially, **format alone earns nothing** —
this removes the incentive to game the output format without solving the problem.

#### Experiments
- **SFT** — accuracy vs. training-set size.
- **GRPO learning rate** — stability vs. learning speed (too-high rates collapse).
- **Baseline** — group-mean baseline as a variance-reduction mechanism.
- **Length normalization** — per-token mean vs. fixed-constant normalization.
- **Std normalization** — keeping vs. dropping the group-std divisor.
        """
    )
    st.caption(f"Model: {D.MODEL} · RL task: Countdown (reach a target number using + − × ÷).")
