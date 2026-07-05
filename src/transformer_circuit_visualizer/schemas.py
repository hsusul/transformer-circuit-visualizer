"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from transformer_circuit_visualizer.config import settings


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"


class ModelMetadata(BaseModel):
    """Basic model metadata needed by the workbench UI."""

    model_name: str
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)


class ModelListResponse(BaseModel):
    """Supported model names and the default model."""

    default_model: str
    models: list[str]


class AnalyzeRequest(BaseModel):
    """Request to run prompt analysis."""

    prompt: str = Field(min_length=1)
    model_name: str = settings.default_model
    top_k: int = Field(default=5, ge=1, le=50)
    attention_layer: int = Field(default=0, ge=0)
    attention_head: int = Field(default=0, ge=0)


class AttentionRequest(BaseModel):
    """Request to retrieve attention data for a single layer and head."""

    prompt: str = Field(min_length=1)
    model_name: str = settings.default_model
    layer: int = Field(ge=0)
    head: int = Field(ge=0)


class HeadSummaryRequest(BaseModel):
    """Request to summarize every attention head for one prompt."""

    prompt: str = Field(min_length=1)
    model_name: str = settings.default_model


class AblateHeadRequest(BaseModel):
    """Request to compare normal predictions with one head ablated."""

    prompt: str = Field(min_length=1)
    model_name: str = settings.default_model
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    top_k: int = Field(default=5, ge=1, le=50)


class TokenPrediction(BaseModel):
    """One next-token prediction."""

    token: str
    token_id: int
    logit: float
    probability: float


class LayerLogitLensResult(BaseModel):
    """Top predictions from one residual-stream layer."""

    layer: int
    top_predictions: list[TokenPrediction]


class AttentionPattern(BaseModel):
    """Attention matrix for one layer/head pair."""

    layer: int
    head: int
    tokens: list[str]
    pattern: list[list[float]]


class HeadSummary(BaseModel):
    """Summary statistics for one attention head."""

    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    average_attention_entropy: float = Field(ge=0)
    max_attention_weight: float = Field(ge=0, le=1)
    previous_token_attention: float = Field(ge=0, le=1)
    bos_attention: float | None = Field(default=None, ge=0, le=1)
    output_norm: float | None = Field(default=None, ge=0)


class PredictionDelta(BaseModel):
    """How one token prediction changed after ablation."""

    token: str
    token_id: int
    before_logit: float
    after_logit: float
    logit_delta: float
    before_probability: float
    after_probability: float
    probability_delta: float


class AnalyzeResponse(BaseModel):
    """Full MVP analysis response."""

    metadata: ModelMetadata
    tokens: list[str]
    token_ids: list[int]
    final_token_predictions: list[TokenPrediction]
    logit_lens: list[LayerLogitLensResult]
    attention: AttentionPattern
    head_summaries: list[HeadSummary]


class AttentionResponse(BaseModel):
    """Standalone attention endpoint response."""

    metadata: ModelMetadata
    attention: AttentionPattern


class HeadSummaryResponse(BaseModel):
    """Standalone head-summary endpoint response."""

    metadata: ModelMetadata
    tokens: list[str]
    head_summaries: list[HeadSummary]


class HeadAblationResponse(BaseModel):
    """Head-ablation comparison response."""

    metadata: ModelMetadata
    layer: int
    head: int
    tokens: list[str]
    before: list[TokenPrediction]
    after: list[TokenPrediction]
    deltas: list[PredictionDelta]
