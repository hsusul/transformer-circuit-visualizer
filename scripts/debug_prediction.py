#!/usr/bin/env python
"""Inspect raw next-token predictions from TransformerLens."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PROMPT = "The capital of France is"
MODEL_NAME = "gpt2-small"
TOKENS_TO_INSPECT = [" Paris", "Paris", " the", ".", "\n"]


def main() -> None:
    """Load GPT-2 small and print raw final-position prediction diagnostics."""

    import torch
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    token_tensor = model.to_tokens(PROMPT, prepend_bos=True)
    logits = model(token_tensor)
    final_logits = logits[0, -1, :]
    probabilities = torch.softmax(final_logits, dim=-1)

    token_ids = [int(token_id) for token_id in token_tensor[0].detach().cpu().tolist()]
    tokens = list(model.to_str_tokens(token_tensor[0]))

    print(f"model: {MODEL_NAME}")
    print(f"prompt: {PROMPT!r}")
    print("\ntokens:")
    for position, (token, token_id) in enumerate(zip(tokens, token_ids, strict=True)):
        print(f"  {position:>2}: id={token_id:<6} token={token!r}")

    print("\nfinal logits top 20 from logits[0, -1, :]:")
    values, top_token_ids = torch.topk(final_logits, k=20)
    for rank, (logit, token_id) in enumerate(zip(values, top_token_ids, strict=True), start=1):
        numeric_token_id = int(token_id.item())
        probability = float(probabilities[numeric_token_id].item())
        decoded = model.to_string(numeric_token_id)
        print(
            f"  {rank:>2}: id={numeric_token_id:<6} "
            f"logit={float(logit.item()):>9.4f} prob={probability:.6f} token={decoded!r}"
        )

    print("\nexplicit token diagnostics:")
    for token_text in TOKENS_TO_INSPECT:
        matching_token_ids = _token_ids_for_text(model, token_text)
        if not matching_token_ids:
            print(f"  {token_text!r}: not a single GPT-2 token")
            continue

        for token_id in matching_token_ids:
            rank = int((final_logits > final_logits[token_id]).sum().item()) + 1
            logit = float(final_logits[token_id].item())
            probability = float(probabilities[token_id].item())
            decoded = model.to_string(token_id)
            print(
                f"  {token_text!r}: id={token_id:<6} rank={rank:<5} "
                f"logit={logit:>9.4f} prob={probability:.6f} decoded={decoded!r}"
            )


def _token_ids_for_text(model: object, token_text: str) -> list[int]:
    try:
        return [int(model.to_single_token(token_text))]
    except Exception:
        tokenizer = getattr(model, "tokenizer")
        encoded = tokenizer.encode(token_text, add_special_tokens=False)
        return [int(token_id) for token_id in encoded] if len(encoded) == 1 else []


if __name__ == "__main__":
    main()
