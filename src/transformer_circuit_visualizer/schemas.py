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


class AnalyzeResponse(BaseModel):
    """Full MVP analysis response."""

    metadata: ModelMetadata
    tokens: list[str]
    token_ids: list[int]
    final_token_predictions: list[TokenPrediction]
    logit_lens: list[LayerLogitLensResult]
    attention: AttentionPattern


class AttentionResponse(BaseModel):
    """Standalone attention endpoint response."""

    metadata: ModelMetadata
    attention: AttentionPattern
