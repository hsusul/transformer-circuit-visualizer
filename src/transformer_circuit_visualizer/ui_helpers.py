"""DataFrame helpers for UI and notebook demos."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def tokens_dataframe(tokens: list[str], token_ids: list[int]) -> pd.DataFrame:
    """Convert token strings and ids into a display table."""

    return pd.DataFrame(
        {
            "position": list(range(len(tokens))),
            "token": tokens,
            "token_id": token_ids,
        }
    )


def predictions_dataframe(predictions: Iterable[Any]) -> pd.DataFrame:
    """Convert top-token predictions into a ranked table."""

    rows = []
    for rank, prediction in enumerate(predictions, start=1):
        item = _as_mapping(prediction)
        rows.append(
            {
                "rank": rank,
                "token": item["token"],
                "token_id": item["token_id"],
                "logit": item["logit"],
                "probability": item["probability"],
            }
        )
    return pd.DataFrame(rows)


def logit_lens_dataframe(layers: Iterable[Any]) -> pd.DataFrame:
    """Flatten per-layer logit lens predictions into one table."""

    rows = []
    for layer_result in layers:
        item = _as_mapping(layer_result)
        for rank, prediction in enumerate(item["top_predictions"], start=1):
            prediction_item = _as_mapping(prediction)
            rows.append(
                {
                    "layer": item["layer"],
                    "rank": rank,
                    "token": prediction_item["token"],
                    "token_id": prediction_item["token_id"],
                    "logit": prediction_item["logit"],
                    "probability": prediction_item["probability"],
                }
            )
    return pd.DataFrame(rows)


def attention_matrix_dataframe(attention: Any) -> pd.DataFrame:
    """Convert an attention pattern into a square matrix DataFrame."""

    item = _as_mapping(attention)
    labels = _token_labels(item["tokens"])
    return pd.DataFrame(item["pattern"], index=labels, columns=labels)


def attention_heatmap_dataframe(attention: Any) -> pd.DataFrame:
    """Convert an attention pattern into long-form heatmap rows."""

    item = _as_mapping(attention)
    rows = []
    for query_position, row in enumerate(item["pattern"]):
        for key_position, weight in enumerate(row):
            rows.append(
                {
                    "query_position": query_position,
                    "key_position": key_position,
                    "query_token": item["tokens"][query_position],
                    "key_token": item["tokens"][key_position],
                    "attention": weight,
                }
            )
    return pd.DataFrame(rows)


def head_summary_dataframe(head_summaries: Iterable[Any]) -> pd.DataFrame:
    """Convert head summary stats into a sortable table."""

    rows = []
    for summary in head_summaries:
        item = _as_mapping(summary)
        rows.append(
            {
                "layer": item["layer"],
                "head": item["head"],
                "average_attention_entropy": item["average_attention_entropy"],
                "max_attention_weight": item["max_attention_weight"],
                "previous_token_attention": item["previous_token_attention"],
                "bos_attention": item["bos_attention"],
                "output_norm": item["output_norm"],
            }
        )
    return pd.DataFrame(rows)


def ablation_delta_dataframe(deltas: Iterable[Any]) -> pd.DataFrame:
    """Convert ablation deltas into a before/after comparison table."""

    rows = []
    for delta in deltas:
        item = _as_mapping(delta)
        rows.append(
            {
                "token": item["token"],
                "token_id": item["token_id"],
                "before_logit": item["before_logit"],
                "after_logit": item["after_logit"],
                "logit_delta": item["logit_delta"],
                "before_probability": item["before_probability"],
                "after_probability": item["after_probability"],
                "probability_delta": item["probability_delta"],
            }
        )
    return pd.DataFrame(rows)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value)!r}.")


def _token_labels(tokens: list[str]) -> list[str]:
    return [f"{index}: {token}" for index, token in enumerate(tokens)]
