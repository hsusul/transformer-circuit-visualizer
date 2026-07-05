"""Tests for UI DataFrame helpers."""

from __future__ import annotations

import pytest

from transformer_circuit_visualizer.analysis import CircuitAnalyzer
from tests.fakes import FakeModelServiceForTests
from transformer_circuit_visualizer.schemas import AnalyzeRequest
from transformer_circuit_visualizer.ui_helpers import (
    ablation_delta_dataframe,
    attention_heatmap_dataframe,
    attention_matrix_dataframe,
    head_summary_dataframe,
    logit_lens_dataframe,
    predictions_dataframe,
    tokens_dataframe,
)


def test_tokens_dataframe_returns_positions_tokens_and_ids() -> None:
    frame = tokens_dataframe(tokens=["<|bos|>", "Hello"], token_ids=[0, 1001])

    assert list(frame.columns) == ["position", "token", "token_id"]
    assert frame.to_dict("records") == [
        {"position": 0, "token": "<|bos|>", "token_id": 0},
        {"position": 1, "token": "Hello", "token_id": 1001},
    ]


def test_predictions_dataframe_accepts_pydantic_models_and_dicts() -> None:
    result = _fake_analysis_result()
    frame_from_models = predictions_dataframe(result.final_token_predictions[:2])
    frame_from_dicts = predictions_dataframe(
        [prediction.model_dump() for prediction in result.final_token_predictions[:2]]
    )

    assert list(frame_from_models.columns) == [
        "rank",
        "token",
        "token_id",
        "logit",
        "probability",
    ]
    assert frame_from_models.equals(frame_from_dicts)
    assert frame_from_models["rank"].tolist() == [1, 2]


def test_logit_lens_dataframe_flattens_layers_and_predictions() -> None:
    result = _fake_analysis_result(top_k=2)

    frame = logit_lens_dataframe(result.logit_lens)

    assert list(frame.columns) == ["layer", "rank", "token", "token_id", "logit", "probability"]
    assert len(frame) == 6
    assert frame[["layer", "rank"]].iloc[0].to_dict() == {"layer": 0, "rank": 1}


def test_attention_dataframe_helpers_return_matrix_and_heatmap_rows() -> None:
    result = _fake_analysis_result()

    matrix = attention_matrix_dataframe(result.attention)
    heatmap = attention_heatmap_dataframe(result.attention)

    assert matrix.shape == (3, 3)
    assert list(matrix.index) == ["0: <|bos|>", "1: Hello", "2: circuits"]
    assert len(heatmap) == 9
    assert set(heatmap.columns) == {
        "query_position",
        "key_position",
        "query_token",
        "key_token",
        "attention",
    }


def test_head_summary_dataframe_returns_one_row_per_head() -> None:
    result = _fake_analysis_result()

    frame = head_summary_dataframe(result.head_summaries)

    assert len(frame) == 6
    assert list(frame.columns) == [
        "layer",
        "head",
        "average_attention_entropy",
        "max_attention_weight",
        "previous_token_attention",
        "bos_attention",
        "output_norm",
    ]
    assert frame.iloc[0]["output_norm"] == 1.0


def test_ablation_delta_dataframe_returns_before_after_columns() -> None:
    service = FakeModelServiceForTests()
    run = service.run_with_cache("fake-gpt2-small", "Hello circuits")
    ablation = service.ablate_head(run, layer=0, head=1, top_k=2)

    frame = ablation_delta_dataframe(ablation.deltas)

    assert len(frame) == 2
    assert list(frame.columns) == [
        "token",
        "token_id",
        "before_logit",
        "after_logit",
        "logit_delta",
        "before_probability",
        "after_probability",
        "probability_delta",
    ]
    assert frame.iloc[0]["logit_delta"] < 0


def test_dataframe_helpers_reject_unknown_objects() -> None:
    with pytest.raises(TypeError):
        predictions_dataframe([object()])


def _fake_analysis_result(top_k: int = 3):
    analyzer = CircuitAnalyzer(FakeModelServiceForTests())
    return analyzer.analyze(
        AnalyzeRequest(prompt="Hello circuits", model_name="fake-gpt2-small", top_k=top_k)
    )
