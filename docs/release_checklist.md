# Release Checklist

Use this checklist before publishing an initial GitHub release.

## Repository

- [ ] Confirm `README.md` has an accurate pitch, screenshot, quickstart, API usage, limitations, and roadmap.
- [ ] Replace the screenshot placeholder with a real demo screenshot.
- [ ] Confirm `LICENSE` is present and correct.
- [ ] Confirm `CONTRIBUTING.md` is present and accurate.
- [ ] Confirm docs are linked and current.

## Validation

- [ ] Run `python -m pytest`.
- [ ] Run `python -m ruff check .`.
- [ ] Run `python scripts/smoke_analyze.py --prompt "The capital of France is"`.
- [ ] Optionally run `python scripts/debug_prediction.py`.

## GitHub

- [ ] Confirm the default branch is clean.
- [ ] Push the release branch.
- [ ] Create a GitHub release tag, for example `v0.1.0`.
- [ ] Include setup notes and known limitations in the release description.
- [ ] Attach or link the demo screenshot.
