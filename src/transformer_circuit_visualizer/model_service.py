"""Model loading and low-level model operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from transformer_circuit_visualizer.config import settings
from transformer_circuit_visualizer.schemas import (
    AttentionPattern,
    ModelMetadata,
    TokenPrediction,
)


@dataclass(frozen=True)
class ModelRun:
    """Raw model outputs from one prompt run."""

    model_name: str
    tokens: list[str]
    token_ids: list[int]
    metadata: ModelMetadata
    logits: Any = None
    cache: Any = None
    model: Any = None


class ModelService(Protocol):
    """Interface used by analysis code and API routes."""

    def list_models(self) -> list[str]:
        """Return model names that the service can load."""

    def run_with_cache(self, model_name: str, prompt: str) -> ModelRun:
        """Tokenize a prompt and run the model with an activation cache."""

    def top_token_predictions(self, run: ModelRun, top_k: int) -> list[TokenPrediction]:
        """Return final-position top-k next-token predictions."""

    def logit_lens(self, run: ModelRun, top_k: int) -> list[list[TokenPrediction]]:
        """Return per-layer final-position logit lens predictions."""

    def attention_pattern(self, run: ModelRun, layer: int, head: int) -> AttentionPattern:
        """Return attention pattern data for one layer/head pair."""


class TransformerLensModelService:
    """Real model service backed by TransformerLens.

    Heavy imports are intentionally delayed until model loading time so tests
    and mock workflows do not require PyTorch or TransformerLens.
    """

    def __init__(
        self,
        model_names: tuple[str, ...] | None = None,
        default_device: str | None = None,
    ) -> None:
        self._model_names = model_names or settings.available_models
        self._default_device = default_device if default_device is not None else settings.device
        self._models: dict[str, Any] = {}

    def list_models(self) -> list[str]:
        """Return configured TransformerLens model names."""

        return list(self._model_names)

    def run_with_cache(self, model_name: str, prompt: str) -> ModelRun:
        """Tokenize a prompt and run the selected model with cache collection."""

        model = self._load_model(model_name)
        token_tensor = model.to_tokens(prompt, prepend_bos=True)
        logits, cache = model.run_with_cache(token_tensor)

        token_ids = [int(token_id) for token_id in token_tensor[0].detach().cpu().tolist()]
        tokens = list(model.to_str_tokens(token_tensor[0]))

        return ModelRun(
            model_name=model_name,
            tokens=tokens,
            token_ids=token_ids,
            metadata=self._metadata(model_name, model),
            logits=logits,
            cache=cache,
            model=model,
        )

    def top_token_predictions(self, run: ModelRun, top_k: int) -> list[TokenPrediction]:
        """Return top-k predictions from the final sequence position."""

        import torch

        final_logits = run.logits[0, -1, :]
        probabilities = torch.softmax(final_logits, dim=-1)
        values, token_ids = torch.topk(final_logits, k=top_k)

        predictions: list[TokenPrediction] = []
        for logit, token_id in zip(values, token_ids, strict=True):
            numeric_token_id = int(token_id.item())
            predictions.append(
                TokenPrediction(
                    token=run.model.to_string(numeric_token_id),
                    token_id=numeric_token_id,
                    logit=float(logit.item()),
                    probability=float(probabilities[numeric_token_id].item()),
                )
            )
        return predictions

    def logit_lens(self, run: ModelRun, top_k: int) -> list[list[TokenPrediction]]:
        """Apply the unembedding to each layer's final-token residual stream."""

        import torch

        results: list[list[TokenPrediction]] = []
        for layer in range(run.metadata.n_layers):
            residual = run.cache["resid_post", layer][:, -1, :]
            normalized = run.model.ln_final(residual)
            layer_logits = run.model.unembed(normalized)[0]
            probabilities = torch.softmax(layer_logits, dim=-1)
            values, token_ids = torch.topk(layer_logits, k=top_k)

            layer_predictions: list[TokenPrediction] = []
            for logit, token_id in zip(values, token_ids, strict=True):
                numeric_token_id = int(token_id.item())
                layer_predictions.append(
                    TokenPrediction(
                        token=run.model.to_string(numeric_token_id),
                        token_id=numeric_token_id,
                        logit=float(logit.item()),
                        probability=float(probabilities[numeric_token_id].item()),
                    )
                )
            results.append(layer_predictions)
        return results

    def attention_pattern(self, run: ModelRun, layer: int, head: int) -> AttentionPattern:
        """Return the cached attention matrix for one layer/head pair."""

        self._validate_layer_head(run.metadata, layer, head)
        pattern = run.cache["pattern", layer][0, head].detach().cpu().tolist()
        return AttentionPattern(layer=layer, head=head, tokens=run.tokens, pattern=pattern)

    def _load_model(self, model_name: str) -> Any:
        if model_name not in self._model_names:
            raise ValueError(f"Unsupported model '{model_name}'.")

        if model_name not in self._models:
            try:
                from transformer_lens import HookedTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "TransformerLens is not installed. Install real analysis dependencies "
                    "with: python -m pip install -e '.[dev]'"
                ) from exc

            kwargs: dict[str, Any] = {}
            if self._default_device:
                kwargs["device"] = self._default_device
            self._models[model_name] = HookedTransformer.from_pretrained(model_name, **kwargs)

        return self._models[model_name]

    @staticmethod
    def _metadata(model_name: str, model: Any) -> ModelMetadata:
        return ModelMetadata(
            model_name=model_name,
            n_layers=int(model.cfg.n_layers),
            n_heads=int(model.cfg.n_heads),
        )

    @staticmethod
    def _validate_layer_head(metadata: ModelMetadata, layer: int, head: int) -> None:
        if layer >= metadata.n_layers:
            raise ValueError(
                f"Layer {layer} is out of range for {metadata.model_name}; "
                f"expected 0-{metadata.n_layers - 1}."
            )
        if head >= metadata.n_heads:
            raise ValueError(
                f"Head {head} is out of range for {metadata.model_name}; "
                f"expected 0-{metadata.n_heads - 1}."
            )


class MockModelService:
    """Deterministic lightweight model service for tests and smoke checks."""

    def __init__(self, model_name: str = "mock-gpt2-small", n_layers: int = 3, n_heads: int = 2) -> None:
        self._model_name = model_name
        self._metadata = ModelMetadata(model_name=model_name, n_layers=n_layers, n_heads=n_heads)
        self._vocab = [" the", " of", " and", " to", ".", " in", " is", " for"]

    def list_models(self) -> list[str]:
        """Return the single mock model name."""

        return [self._model_name]

    def run_with_cache(self, model_name: str, prompt: str) -> ModelRun:
        """Create deterministic token data without loading a real model."""

        if model_name != self._model_name:
            raise ValueError(f"Unsupported mock model '{model_name}'.")

        pieces = prompt.strip().split()
        tokens = ["<|bos|>"] + (pieces if pieces else [prompt])
        token_ids = [0] + [1000 + index for index, _ in enumerate(tokens[1:], start=1)]

        return ModelRun(
            model_name=model_name,
            tokens=tokens,
            token_ids=token_ids,
            metadata=self._metadata,
        )

    def top_token_predictions(self, run: ModelRun, top_k: int) -> list[TokenPrediction]:
        """Return deterministic final-token predictions."""

        return self._predictions(top_k=top_k, offset=0)

    def logit_lens(self, run: ModelRun, top_k: int) -> list[list[TokenPrediction]]:
        """Return deterministic per-layer logit lens predictions."""

        return [self._predictions(top_k=top_k, offset=layer) for layer in range(run.metadata.n_layers)]

    def attention_pattern(self, run: ModelRun, layer: int, head: int) -> AttentionPattern:
        """Return a simple causal attention pattern for one layer/head."""

        TransformerLensModelService._validate_layer_head(run.metadata, layer, head)
        token_count = len(run.tokens)
        pattern: list[list[float]] = []

        for destination in range(token_count):
            allowed = destination + 1
            row = [0.0] * token_count
            for source in range(allowed):
                row[source] = round(1.0 / allowed, 6)
            pattern.append(row)

        return AttentionPattern(layer=layer, head=head, tokens=run.tokens, pattern=pattern)

    def _predictions(self, top_k: int, offset: int) -> list[TokenPrediction]:
        predictions: list[TokenPrediction] = []
        for rank in range(top_k):
            vocab_index = (rank + offset) % len(self._vocab)
            probability = 1.0 / float(rank + 2)
            predictions.append(
                TokenPrediction(
                    token=self._vocab[vocab_index],
                    token_id=2000 + vocab_index,
                    logit=10.0 - rank - (offset * 0.1),
                    probability=probability,
                )
            )
        return predictions
