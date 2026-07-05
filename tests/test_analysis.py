"""Analysis workflow tests using the test fake model service."""

from __future__ import annotations

from transformer_circuit_visualizer.analysis import CircuitAnalyzer
from tests.fakes import FakeModelServiceForTests
from transformer_circuit_visualizer.schemas import (
    AblateHeadRequest,
    AnalyzeRequest,
    AttentionRequest,
    HeadSummaryRequest,
)


def test_analyzer_returns_tokenization_predictions_and_attention() -> None:
    analyzer = CircuitAnalyzer(FakeModelServiceForTests())

    result = analyzer.analyze(
        AnalyzeRequest(prompt="Mechanistic interpretability", model_name="fake-gpt2-small", top_k=2)
    )

    assert result.metadata.model_name == "fake-gpt2-small"
    assert result.metadata.n_layers == 3
    assert result.metadata.n_heads == 2
    assert result.tokens == ["<|bos|>", "Mechanistic", "interpretability"]
    assert result.token_ids == [0, 1001, 1002]
    assert [prediction.token for prediction in result.final_token_predictions] == [" the", " of"]
    assert len(result.logit_lens) == 3
    assert all(len(layer.top_predictions) == 2 for layer in result.logit_lens)
    assert result.attention.pattern[0] == [1.0, 0.0, 0.0]
    assert len(result.head_summaries) == 6


def test_analyzer_returns_requested_attention_layer_and_head() -> None:
    analyzer = CircuitAnalyzer(FakeModelServiceForTests())

    result = analyzer.attention(
        AttentionRequest(
            prompt="Attention check",
            model_name="fake-gpt2-small",
            layer=2,
            head=1,
        )
    )

    assert result.attention.layer == 2
    assert result.attention.head == 1
    assert result.attention.tokens == ["<|bos|>", "Attention", "check"]


def test_analyzer_returns_head_summary() -> None:
    analyzer = CircuitAnalyzer(FakeModelServiceForTests())

    result = analyzer.head_summary(
        HeadSummaryRequest(prompt="Head summary", model_name="fake-gpt2-small")
    )

    assert result.metadata.n_layers == 3
    assert result.metadata.n_heads == 2
    assert result.tokens == ["<|bos|>", "Head", "summary"]
    assert len(result.head_summaries) == 6
    assert result.head_summaries[1].previous_token_attention == 0.3


def test_analyzer_returns_head_ablation_comparison() -> None:
    analyzer = CircuitAnalyzer(FakeModelServiceForTests())

    result = analyzer.ablate_head(
        AblateHeadRequest(prompt="Ablate this", model_name="fake-gpt2-small", layer=0, head=1, top_k=2)
    )

    assert result.layer == 0
    assert result.head == 1
    assert len(result.before) == 2
    assert len(result.after) == 2
    assert len(result.deltas) == 2
    assert result.deltas[0].probability_delta < 0
