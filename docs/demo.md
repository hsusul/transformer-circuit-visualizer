# Demo Guide

This project includes a lightweight Streamlit demo for local mechanistic interpretability exploration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ui,test]"
```

## Run

```bash
streamlit run apps/streamlit_app.py
```

Open `http://localhost:8501`.

## Suggested Walkthrough

1. Keep the default prompt, `The capital of France is`.
2. Select `gpt2-small`.
3. Set top-k to 5 or 10.
4. Run analysis.
5. Inspect tokenization first. GPT-2 tokens often include leading spaces.
6. Inspect final next-token predictions. These are raw continuations, not answers from a QA system.
7. Open the logit lens table and compare how top predictions shift across layers.
8. Select an attention layer/head and inspect the heatmap.
9. Review the head summary table for high BOS attention or low-entropy heads.
10. Try a single-head ablation and compare before/after predictions.

## Notes

- First model load can take time while weights download and initialize.
- CPU inference may be slow. Use `TCV_DEVICE=cpu` only when automatic device selection fails.
- If `gpt2-small` is too heavy for a quick check, try `attn-only-1l`.
