# Roadmap

This roadmap is intentionally scoped around making the project useful without hiding model behavior behind abstractions.

## Near Term

- Replace the screenshot placeholder with a real demo screenshot.
- Add clearer progress indicators for model loading and analysis.
- Improve model selection metadata: parameter count, architecture, and expected memory cost.
- Add compact response modes for CLI and API debugging.
- Add more focused tests around cache keys and logit lens behavior.

## Interpretability Workbench

- Add residual stream views.
- Add MLP/neuron inspection.
- Add prompt comparison workflows.
- Add attention-head search and filtering.
- Add richer ablation controls for layers, heads, and components.
- Add exportable analysis bundles.

## Frontend

- Keep Streamlit as the demo UI while backend contracts evolve.
- Build the production frontend later with Next.js.
- Add visual comparison layouts for tokens, heads, layers, and ablations.
- Add shareable local reports.

## Engineering

- Add structured logging.
- Add better error handling for model download, device, and memory failures.
- Add optional local model cache controls.
- Add CI once dependency/runtime costs are clear.
- Add release automation after the first stable tag.
