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
