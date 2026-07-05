"""FastAPI application and routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from transformer_circuit_visualizer.analysis import CircuitAnalyzer
from transformer_circuit_visualizer.config import settings
from transformer_circuit_visualizer.model_service import (
    ModelService,
    TransformerLensModelService,
)
from transformer_circuit_visualizer.schemas import (
    AblateHeadRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    AttentionRequest,
    AttentionResponse,
    HeadAblationResponse,
    HeadSummaryRequest,
    HeadSummaryResponse,
    HealthResponse,
    ModelListResponse,
)


def create_app(model_service: ModelService | None = None) -> FastAPI:
    """Create the FastAPI app with an injectable model service."""

    app = FastAPI(
        title="Transformer Circuit Visualizer",
        version="0.1.0",
        description="Mechanistic interpretability workbench backend.",
    )
    app.state.model_service = model_service or TransformerLensModelService()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return process health."""

        return HealthResponse()

    @app.get("/models", response_model=ModelListResponse)
    def models(request: Request) -> ModelListResponse:
        """Return configured model names."""

        service: ModelService = request.app.state.model_service
        return ModelListResponse(default_model=settings.default_model, models=service.list_models())

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        """Run tokenization, final predictions, logit lens, and attention analysis."""

        analyzer = CircuitAnalyzer(request.app.state.model_service)
        try:
            return analyzer.analyze(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/attention", response_model=AttentionResponse)
    def attention(payload: AttentionRequest, request: Request) -> AttentionResponse:
        """Return attention heatmap data for one layer and head."""

        analyzer = CircuitAnalyzer(request.app.state.model_service)
        try:
            return analyzer.attention(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/heads/summary", response_model=HeadSummaryResponse)
    def heads_summary(payload: HeadSummaryRequest, request: Request) -> HeadSummaryResponse:
        """Return summary statistics for every attention head."""

        analyzer = CircuitAnalyzer(request.app.state.model_service)
        try:
            return analyzer.head_summary(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/ablate/head", response_model=HeadAblationResponse)
    def ablate_head(payload: AblateHeadRequest, request: Request) -> HeadAblationResponse:
        """Compare final predictions before and after ablating one head."""

        analyzer = CircuitAnalyzer(request.app.state.model_service)
        try:
            return analyzer.ablate_head(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app
