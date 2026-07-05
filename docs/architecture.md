# Architecture

`transformer-circuit-visualizer` is intentionally small. The current architecture separates API routing, model loading, analysis orchestration, schemas, and UI helper formatting.

## Modules

- `src/transformer_circuit_visualizer/config.py`: runtime settings and configured model names.
- `src/transformer_circuit_visualizer/schemas.py`: Pydantic request and response models.
- `src/transformer_circuit_visualizer/model_service.py`: TransformerLens model loading, tokenization, activation cache collection, predictions, logit lens, attention patterns, head summaries, and ablations.
- `src/transformer_circuit_visualizer/analysis.py`: high-level workflows used by API routes and scripts.
- `src/transformer_circuit_visualizer/api.py`: FastAPI app factory and route handlers.
- `src/transformer_circuit_visualizer/main.py`: ASGI app entrypoint and local server runner.
- `src/transformer_circuit_visualizer/ui_helpers.py`: conversion helpers for Streamlit tables and charts.
- `apps/streamlit_app.py`: local demo UI.
- `scripts/smoke_analyze.py`: one-prompt smoke analysis.
- `scripts/debug_prediction.py`: raw next-token prediction debugger.

## Request Flow

1. A caller sends a prompt and model name to the API, CLI, or Streamlit UI.
2. `CircuitAnalyzer` asks `TransformerLensModelService` to run the model with cache.
3. The service tokenizes the prompt with `prepend_bos=True`.
4. TransformerLens returns logits and an activation cache.
5. Analysis code builds typed responses:
   - final next-token predictions from `logits[0, -1, :]`
   - logit lens values from final-position residual streams
   - attention pattern matrices from cached attention patterns
   - per-head summary statistics
   - optional single-head ablation comparisons

## Model Service Boundary

Model loading is isolated in `model_service.py` so future work can add caching, progress reporting, device policy, or alternate backends without changing route handlers.

Tests use `tests/fakes.py` for deterministic unit coverage. The application package itself defaults to the real TransformerLens service.

## Current Non-Goals

- No persistence layer.
- No authentication.
- No background job queue.
- No production frontend.
- No distributed model serving.
