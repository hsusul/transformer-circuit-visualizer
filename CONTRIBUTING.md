# Contributing

Thanks for considering a contribution to `transformer-circuit-visualizer`.

## Setup

```bash
git clone https://github.com/hsusul/transformer-circuit-visualizer.git
cd transformer-circuit-visualizer
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ui,test]"
```

The project uses real TransformerLens and PyTorch dependencies by default. The first run may download model weights.

## Tests

Run the standard test suite:

```bash
python -m pytest
```

Run lint:

```bash
python -m ruff check .
```

Run a local smoke analysis:

```bash
python scripts/smoke_analyze.py --prompt "The capital of France is"
```

Most tests use deterministic fakes so contributors can validate changes without repeated model downloads. Use `scripts/debug_prediction.py` when investigating raw TransformerLens prediction behavior.

## Code Style

- Keep modules small and explicit.
- Keep model loading inside `model_service.py`.
- Keep analysis orchestration inside `analysis.py`.
- Keep FastAPI route handling inside `api.py`.
- Use Pydantic schemas for API shapes.
- Prefer focused tests with deterministic fakes for normal unit coverage.
- Do not add databases, auth, background jobs, Docker, or frontend framework changes without opening an issue first.

## Suggested Contribution Areas

- Better TransformerLens cache-key compatibility across supported models.
- More robust logit lens and residual-stream diagnostics.
- UI helper improvements for clearer tables and charts.
- Documentation examples with real screenshots.
- Smaller, reliable test models for local validation.
- Better error messages for device, dtype, memory, and download failures.
