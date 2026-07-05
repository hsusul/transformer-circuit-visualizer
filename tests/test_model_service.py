"""Model service tests."""

from __future__ import annotations

import pytest

from transformer_circuit_visualizer.model_service import MockModelService


def test_mock_model_service_lists_model() -> None:
    service = MockModelService()

    assert service.list_models() == ["mock-gpt2-small"]


def test_mock_model_service_rejects_unknown_model() -> None:
    service = MockModelService()

    with pytest.raises(ValueError, match="Unsupported mock model"):
        service.run_with_cache("unknown-model", "Hello")


def test_mock_attention_is_causal_and_square() -> None:
    service = MockModelService()
    run = service.run_with_cache("mock-gpt2-small", "a b c")

    attention = service.attention_pattern(run, layer=0, head=0)

    assert len(attention.pattern) == len(run.tokens)
    assert all(len(row) == len(run.tokens) for row in attention.pattern)
    assert attention.pattern[0] == [1.0, 0.0, 0.0, 0.0]
    assert attention.pattern[-1] == [0.25, 0.25, 0.25, 0.25]


def test_mock_head_summary_returns_one_row_per_head() -> None:
    service = MockModelService()
    run = service.run_with_cache("mock-gpt2-small", "a b")

    summaries = service.head_summary(run)

    assert len(summaries) == 6
    assert summaries[0].layer == 0
    assert summaries[0].head == 0
    assert summaries[0].max_attention_weight == 1.0
    assert summaries[0].bos_attention is not None
    assert summaries[0].output_norm == 1.0
    assert summaries[-1].layer == 2
    assert summaries[-1].head == 1


def test_mock_ablation_returns_before_after_and_deltas() -> None:
    service = MockModelService()
    run = service.run_with_cache("mock-gpt2-small", "a b")

    result = service.ablate_head(run, layer=1, head=1, top_k=2)

    assert result.layer == 1
    assert result.head == 1
    assert len(result.before) == 2
    assert len(result.after) == 2
    assert len(result.deltas) == 2
    assert result.after[0].logit < result.before[0].logit
    assert result.deltas[0].logit_delta < 0
