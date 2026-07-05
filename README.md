# transformer-circuit-visualizer

An open-source mechanistic interpretability workbench for transformer models.

The MVP is a Python/FastAPI backend. It accepts a prompt, runs a small
TransformerLens-supported model, and returns tokenization, next-token
predictions, layer-by-layer logit lens predictions, and attention heatmap data.

## Current MVP

- FastAPI backend
- Optional TransformerLens model service
- Mock model service for local tests and lightweight smoke checks
- Pydantic request/response schemas
- pytest coverage for schemas, API health, analysis, and mock service behavior
- Per-head attention summaries
- Single-head ablation comparisons

No frontend, database, auth, background jobs, or Docker are included yet.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the lightweight API/test dependencies:

```bash
python -m pip install -e ".[test]"
```

Install real model-analysis dependencies when you are ready to use
TransformerLens:

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

## Smoke Check

The default smoke script uses the mock model service, so it does not download
model weights:

```bash
python scripts/smoke_analyze.py
```

To use the real TransformerLens service after installing the `dev` extra:

```bash
python scripts/smoke_analyze.py --real --prompt "The capital of France is"
```

## Run API

```bash
uvicorn transformer_circuit_visualizer.main:app --reload
```

Then open:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/models`

Example analysis request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","model_name":"gpt2-small","top_k":5}'
```

Example head summary request:

```bash
curl -X POST http://127.0.0.1:8000/heads/summary \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","model_name":"gpt2-small"}'
```

Example head ablation request:

```bash
curl -X POST http://127.0.0.1:8000/ablate/head \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","model_name":"gpt2-small","layer":0,"head":0,"top_k":5}'
```

## API Shape

- `GET /health`
- `GET /models`
- `POST /analyze`
- `POST /attention`
- `POST /heads/summary`
- `POST /ablate/head`

## Phase 2 Analysis

Head summaries return one row per layer/head with:

- average attention entropy
- max attention weight
- average attention paid to the previous token
- average attention paid to the BOS/start token when present
- output norm when available from the activation cache

Head ablation compares final-token predictions before and after zeroing one
attention head's value stream. The response includes before/after top-k
predictions plus logit and probability deltas for the reported tokens.

The default app uses the real TransformerLens service. Tests and the smoke
script can inject `MockModelService` to avoid downloads.
