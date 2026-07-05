"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformer_circuit_visualizer.schemas import (
    AblateHeadRequest,
    AnalyzeRequest,
    AttentionRequest,
    HeadSummary,
    HeadSummaryRequest,
    ModelMetadata,
)


def test_analyze_request_defaults() -> None:
    request = AnalyzeRequest(prompt="Hello world")

    assert request.model_name == "gpt2-small"
    assert request.top_k == 5
    assert request.attention_layer == 0
    assert request.attention_head == 0


def test_analyze_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(prompt="")


def test_analyze_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(prompt="Hello", top_k=0)

    with pytest.raises(ValidationError):
        AnalyzeRequest(prompt="Hello", top_k=51)


def test_attention_request_requires_non_negative_layer_and_head() -> None:
    with pytest.raises(ValidationError):
        AttentionRequest(prompt="Hello", layer=-1, head=0)

    with pytest.raises(ValidationError):
        AttentionRequest(prompt="Hello", layer=0, head=-1)


def test_metadata_requires_positive_dimensions() -> None:
    with pytest.raises(ValidationError):
        ModelMetadata(model_name="bad", n_layers=0, n_heads=1)


def test_head_summary_request_defaults_to_configured_model() -> None:
    request = HeadSummaryRequest(prompt="Hello world")

    assert request.model_name == "gpt2-small"


def test_head_summary_validates_probability_like_fields() -> None:
    summary = HeadSummary(
        layer=0,
        head=1,
        average_attention_entropy=0.5,
        max_attention_weight=1.0,
        previous_token_attention=0.25,
        bos_attention=0.4,
        output_norm=2.0,
    )

    assert summary.layer == 0
    assert summary.head == 1

    with pytest.raises(ValidationError):
        HeadSummary(
            layer=0,
            head=0,
            average_attention_entropy=0.0,
            max_attention_weight=1.5,
            previous_token_attention=0.0,
        )


def test_ablate_head_request_validates_top_k() -> None:
    request = AblateHeadRequest(prompt="Hello", layer=0, head=0, top_k=3)

    assert request.top_k == 3

    with pytest.raises(ValidationError):
        AblateHeadRequest(prompt="Hello", layer=0, head=0, top_k=0)
