# Math Reasoning with Reinforcement Learning

Interactive dashboard for a project that improves a 1.5B language model's
mathematical reasoning with **supervised finetuning (SFT)** and **Group-Relative
Policy Optimization (GRPO)**, on `Qwen2.5-Math-1.5B`.

## Run

```sh
uv run --with streamlit streamlit run dashboard/app.py
```

Opens at http://localhost:8501. No GPU required — the reward function is pure Python.

## Tabs

| Tab | What |
|---|---|
| **Overview** | Pipeline diagram + headline results |
| **🎮 Reward Playground** | *Live.* Type a model response + target answer and watch the math reward function score it — checks mathematical equivalence of fractions, decimals, and LaTeX. Runs in-browser on CPU. |
| **Baseline** | Zero-shot accuracy + a breakdown of *why* the base model fails |
| **SFT** | How accuracy scales with training-set size |
| **GRPO** | Reinforcement-learning reward curve + four ablations (learning rate, baseline, length norm, std norm) and example responses |
| **📐 Approach** | The methods: evaluation, SFT, GRPO, and reward design |

## Files

- `app.py` — the Streamlit app
- `data.py` — all result numbers (single source of truth)
- `assets/` — result figures

To update numbers, edit `data.py` only.
