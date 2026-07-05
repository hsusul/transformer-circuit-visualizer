"""Opt-in smoke test for the real TransformerLens integration."""

from __future__ import annotations

import os
from typing import Any

import pytest

from transformer_circuit_visualizer.analysis import CircuitAnalyzer
from transformer_circuit_visualizer.model_service import TransformerLensModelService
from transformer_circuit_visualizer.schemas import AnalyzeRequest


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to run real TransformerLens model smoke tests.",
)
def test_real_transformer_lens_analyze_smoke() -> None:
    """Run one real prompt through tokenization, cache collection, and summaries."""

    model_name = os.getenv("TCV_REAL_TEST_MODEL", "gpt2-small")
    analyzer = CircuitAnalyzer(TransformerLensModelService(model_names=(model_name,)))

    result = analyzer.analyze(
        AnalyzeRequest(
            prompt="The capital of France is",
            model_name=model_name,
            top_k=3,
        )
    )

    assert result.metadata.model_name == model_name
    assert result.metadata.n_layers > 0
    assert result.metadata.n_heads > 0
    assert result.tokens
    assert result.token_ids
    assert len(result.final_token_predictions) == 3
    assert result.logit_lens
    assert result.attention.pattern
    assert result.head_summaries


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to run real TransformerLens model smoke tests.",
)
def test_gpt2_small_paris_appears_in_raw_top_20_next_token_predictions() -> None:
    """Lock the expected raw final-logit behavior for the France prompt."""

    import torch
    from transformer_lens import HookedTransformer

    prompt = "The capital of France is"
    model = HookedTransformer.from_pretrained("gpt2-small")
    token_tensor = model.to_tokens(prompt, prepend_bos=True)
    logits = model(token_tensor)
    final_logits = logits[0, -1, :]
    probabilities = torch.softmax(final_logits, dim=-1)
    top_token_ids = torch.topk(final_logits, k=20).indices
    top_tokens = [model.to_string(int(token_id.item())) for token_id in top_token_ids]

    paris_token_id = int(model.to_single_token(" Paris"))
    paris_rank = int((final_logits > final_logits[paris_token_id]).sum().item()) + 1
    diagnostics = _prediction_diagnostics(
        model=model,
        token_tensor=token_tensor,
        final_logits=final_logits,
        probabilities=probabilities,
        top_token_ids=top_token_ids,
        paris_token_id=paris_token_id,
        paris_rank=paris_rank,
    )

    assert " Paris" in top_tokens, diagnostics


def _prediction_diagnostics(
    model: Any,
    token_tensor: Any,
    final_logits: Any,
    probabilities: Any,
    top_token_ids: Any,
    paris_token_id: int,
    paris_rank: int,
) -> str:
    tokens = list(model.to_str_tokens(token_tensor[0]))
    token_ids = [int(token_id) for token_id in token_tensor[0].detach().cpu().tolist()]
    top_20 = [
        {
            "rank": rank,
            "token": model.to_string(int(token_id.item())),
            "token_id": int(token_id.item()),
            "logit": float(final_logits[int(token_id.item())].item()),
            "probability": float(probabilities[int(token_id.item())].item()),
        }
        for rank, token_id in enumerate(top_token_ids, start=1)
    ]

    return (
        "' Paris' was not in the raw top 20 from logits[0, -1, :]. "
        f"Tokens={list(enumerate(zip(tokens, token_ids, strict=True)))}. "
        f"Paris token id={paris_token_id}, rank={paris_rank}, "
        f"logit={float(final_logits[paris_token_id].item())}, "
        f"probability={float(probabilities[paris_token_id].item())}. "
        f"Top 20={top_20}"
    )
