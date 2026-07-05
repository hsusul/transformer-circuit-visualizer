"""Model loading and low-level model operations."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Protocol

from transformer_circuit_visualizer.config import settings
from transformer_circuit_visualizer.schemas import (
    HeadAblationResponse,
    HeadSummary,
    AttentionPattern,
    ModelMetadata,
    PredictionDelta,
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
    token_tensor: Any = None


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

    def head_summary(self, run: ModelRun) -> list[HeadSummary]:
        """Return summary statistics for every layer/head pair."""

    def ablate_head(
        self,
        run: ModelRun,
        layer: int,
        head: int,
        top_k: int,
    ) -> HeadAblationResponse:
        """Compare final-token predictions before and after one head is ablated."""


class TransformerLensModelService:
    """Real model service backed by TransformerLens.

    Imports are checked at model loading time so dependency errors can explain
    the install command before TransformerLens emits a lower-level traceback.
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
            token_tensor=token_tensor,
        )

    def top_token_predictions(self, run: ModelRun, top_k: int) -> list[TokenPrediction]:
        """Return top-k predictions from the final sequence position."""

        return self._predictions_from_logits(run, run.logits[0, -1, :], top_k)

    def head_summary(self, run: ModelRun) -> list[HeadSummary]:
        """Summarize attention behavior for every head."""

        import torch

        summaries: list[HeadSummary] = []
        for layer in range(run.metadata.n_layers):
            patterns = run.cache["pattern", layer][0].detach()
            z_values = self._head_outputs(run, layer)

            for head in range(run.metadata.n_heads):
                pattern = patterns[head]
                clamped = pattern.clamp_min(1e-12)
                entropy = -(clamped * torch.log(clamped)).sum(dim=-1).mean()

                previous_token_attention = 0.0
                if pattern.shape[0] > 1:
                    positions = torch.arange(1, pattern.shape[0], device=pattern.device)
                    previous_token_attention = float(pattern[positions, positions - 1].mean().item())

                output_norm = None
                if z_values is not None:
                    output_norm = float(z_values[:, head, :].norm(dim=-1).mean().item())

                summaries.append(
                    HeadSummary(
                        layer=layer,
                        head=head,
                        average_attention_entropy=float(entropy.item()),
                        max_attention_weight=float(pattern.max().item()),
                        previous_token_attention=previous_token_attention,
                        bos_attention=self._bos_attention(run, pattern),
                        output_norm=output_norm,
                    )
                )
        return summaries

    def ablate_head(
        self,
        run: ModelRun,
        layer: int,
        head: int,
        top_k: int,
    ) -> HeadAblationResponse:
        """Zero one head's value stream and compare final-token predictions."""

        self._validate_layer_head(run.metadata, layer, head)
        ablated_logits = self._run_with_head_ablation(run, layer, head)

        before_logits = run.logits[0, -1, :]
        after_logits = ablated_logits[0, -1, :]
        before = self._predictions_from_logits(run, before_logits, top_k)
        after = self._predictions_from_logits(run, after_logits, top_k)

        return HeadAblationResponse(
            metadata=run.metadata,
            layer=layer,
            head=head,
            tokens=run.tokens,
            before=before,
            after=after,
            deltas=self._prediction_deltas(run, before_logits, after_logits, before, after),
        )

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

    def _predictions_from_logits(
        self,
        run: ModelRun,
        logits: Any,
        top_k: int,
    ) -> list[TokenPrediction]:
        import torch

        probabilities = torch.softmax(logits, dim=-1)
        values, token_ids = torch.topk(logits, k=top_k)

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

    def _prediction_deltas(
        self,
        run: ModelRun,
        before_logits: Any,
        after_logits: Any,
        before: list[TokenPrediction],
        after: list[TokenPrediction],
    ) -> list[PredictionDelta]:
        import torch

        before_probabilities = torch.softmax(before_logits, dim=-1)
        after_probabilities = torch.softmax(after_logits, dim=-1)
        token_ids = self._ordered_prediction_token_ids(before, after)

        deltas: list[PredictionDelta] = []
        for token_id in token_ids:
            before_logit = float(before_logits[token_id].item())
            after_logit = float(after_logits[token_id].item())
            before_probability = float(before_probabilities[token_id].item())
            after_probability = float(after_probabilities[token_id].item())
            deltas.append(
                PredictionDelta(
                    token=run.model.to_string(token_id),
                    token_id=token_id,
                    before_logit=before_logit,
                    after_logit=after_logit,
                    logit_delta=after_logit - before_logit,
                    before_probability=before_probability,
                    after_probability=after_probability,
                    probability_delta=after_probability - before_probability,
                )
            )
        return deltas

    def _run_with_head_ablation(self, run: ModelRun, layer: int, head: int) -> Any:
        def ablate_z(value: Any, hook: Any) -> Any:
            del hook
            ablated = value.clone()
            ablated[:, :, head, :] = 0.0
            return ablated

        return run.model.run_with_hooks(
            run.token_tensor,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablate_z)],
        )

    @staticmethod
    def _head_outputs(run: ModelRun, layer: int) -> Any | None:
        try:
            return run.cache["z", layer][0].detach()
        except KeyError:
            return None

    @staticmethod
    def _bos_attention(run: ModelRun, pattern: Any) -> float | None:
        if not run.tokens:
            return None

        first_token = run.tokens[0].lower()
        start_like = (
            "bos" in first_token
            or "start" in first_token
            or "endoftext" in first_token
            or first_token == "<s>"
        )

        tokenizer = getattr(run.model, "tokenizer", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        known_start_id = run.token_ids[0] in {bos_token_id, eos_token_id}

        if not (start_like or known_start_id):
            return None
        return float(pattern[:, 0].mean().item())

    @staticmethod
    def _ordered_prediction_token_ids(
        before: list[TokenPrediction],
        after: list[TokenPrediction],
    ) -> list[int]:
        token_ids: list[int] = []
        seen: set[int] = set()
        for prediction in [*before, *after]:
            if prediction.token_id not in seen:
                seen.add(prediction.token_id)
                token_ids.append(prediction.token_id)
        return token_ids

    def _load_model(self, model_name: str) -> Any:
        if model_name not in self._model_names:
            raise ValueError(f"Unsupported model '{model_name}'.")

        if model_name not in self._models:
            self._check_real_dependencies()
            try:
                from transformer_lens import HookedTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "TransformerLens could not be imported. Install the project runtime "
                    "dependencies with `python -m pip install -e .`, or install the "
                    "full local environment with "
                    "`python -m pip install -e '.[dev,ui,test]'`."
                ) from exc

            kwargs: dict[str, Any] = {}
            if self._default_device:
                kwargs["device"] = self._default_device
            self._models[model_name] = HookedTransformer.from_pretrained(model_name, **kwargs)

        return self._models[model_name]

    @staticmethod
    def _check_real_dependencies() -> None:
        required_modules = {
            "torch": "torch",
            "transformer_lens": "transformer-lens",
            "einops": "einops",
        }
        missing = [
            package_name
            for module_name, package_name in required_modules.items()
            if importlib.util.find_spec(module_name) is None
        ]
        if missing:
            missing_names = ", ".join(missing)
            raise RuntimeError(
                "Transformer Circuit Visualizer requires real TransformerLens runtime "
                f"dependencies, but these packages are missing: {missing_names}. "
                "Install them with `python -m pip install -e .`, or install the full "
                "local environment with `python -m pip install -e '.[dev,ui,test]'`."
            )

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

