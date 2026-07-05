"""Prompt analysis orchestration."""

from __future__ import annotations

from transformer_circuit_visualizer.model_service import ModelService
from transformer_circuit_visualizer.schemas import (
    AblateHeadRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    AttentionRequest,
    AttentionResponse,
    HeadAblationResponse,
    HeadSummaryRequest,
    HeadSummaryResponse,
    LayerLogitLensResult,
)


class CircuitAnalyzer:
    """High-level analysis workflow used by API routes and scripts."""

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Run the complete MVP analysis for one prompt."""

        run = self._model_service.run_with_cache(request.model_name, request.prompt)
        logit_lens = [
            LayerLogitLensResult(layer=layer, top_predictions=predictions)
            for layer, predictions in enumerate(
                self._model_service.logit_lens(run, top_k=request.top_k)
            )
        ]

        return AnalyzeResponse(
            metadata=run.metadata,
            tokens=run.tokens,
            token_ids=run.token_ids,
            final_token_predictions=self._model_service.top_token_predictions(
                run, top_k=request.top_k
            ),
            logit_lens=logit_lens,
            attention=self._model_service.attention_pattern(
                run,
                layer=request.attention_layer,
                head=request.attention_head,
            ),
            head_summaries=self._model_service.head_summary(run),
        )

    def attention(self, request: AttentionRequest) -> AttentionResponse:
        """Return only attention data for one prompt and layer/head pair."""

        run = self._model_service.run_with_cache(request.model_name, request.prompt)
        return AttentionResponse(
            metadata=run.metadata,
            attention=self._model_service.attention_pattern(
                run,
                layer=request.layer,
                head=request.head,
            ),
        )

    def head_summary(self, request: HeadSummaryRequest) -> HeadSummaryResponse:
        """Return per-layer/per-head summary statistics."""

        run = self._model_service.run_with_cache(request.model_name, request.prompt)
        return HeadSummaryResponse(
            metadata=run.metadata,
            tokens=run.tokens,
            head_summaries=self._model_service.head_summary(run),
        )

    def ablate_head(self, request: AblateHeadRequest) -> HeadAblationResponse:
        """Compare normal predictions with one ablated attention head."""

        run = self._model_service.run_with_cache(request.model_name, request.prompt)
        return self._model_service.ablate_head(
            run,
            layer=request.layer,
            head=request.head,
            top_k=request.top_k,
        )
