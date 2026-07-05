"""Test-only fakes for fast, deterministic unit tests."""

from __future__ import annotations

import math

from transformer_circuit_visualizer.model_service import (
    ModelRun,
    TransformerLensModelService,
)
from transformer_circuit_visualizer.schemas import (
    AttentionPattern,
    HeadAblationResponse,
    HeadSummary,
    ModelMetadata,
    PredictionDelta,
    TokenPrediction,
)


class FakeModelServiceForTests:
    """Deterministic model service used only by tests."""

    def __init__(
        self,
        model_name: str = "fake-gpt2-small",
        n_layers: int = 3,
        n_heads: int = 2,
    ) -> None:
        self._model_name = model_name
        self._metadata = ModelMetadata(model_name=model_name, n_layers=n_layers, n_heads=n_heads)
        self._vocab = [" the", " of", " and", " to", ".", " in", " is", " for"]

    def list_models(self) -> list[str]:
        """Return the single fake model name."""

        return [self._model_name]

    def run_with_cache(self, model_name: str, prompt: str) -> ModelRun:
        """Create deterministic token data without loading a real model."""

        if model_name != self._model_name:
            raise ValueError(f"Unsupported test fake model '{model_name}'.")

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
        pattern = self._attention_matrix(len(run.tokens), head=head)

        return AttentionPattern(layer=layer, head=head, tokens=run.tokens, pattern=pattern)

    def head_summary(self, run: ModelRun) -> list[HeadSummary]:
        """Return deterministic head summaries."""

        summaries: list[HeadSummary] = []
        for layer in range(run.metadata.n_layers):
            for head in range(run.metadata.n_heads):
                pattern = self._attention_matrix(len(run.tokens), head=head)
                summaries.append(
                    HeadSummary(
                        layer=layer,
                        head=head,
                        average_attention_entropy=self._average_entropy(pattern),
                        max_attention_weight=max(max(row) for row in pattern),
                        previous_token_attention=self._previous_token_attention(pattern),
                        bos_attention=sum(row[0] for row in pattern) / len(pattern),
                        output_norm=round(1.0 + layer + (head * 0.1), 6),
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
        """Return deterministic before/after predictions for a fake head ablation."""

        TransformerLensModelService._validate_layer_head(run.metadata, layer, head)
        before = self.top_token_predictions(run, top_k=top_k)
        penalty = 0.5 + (layer * 0.1) + (head * 0.05)
        after = [
            TokenPrediction(
                token=prediction.token,
                token_id=prediction.token_id,
                logit=prediction.logit - penalty,
                probability=max(prediction.probability - 0.05 * (rank + 1), 0.0),
            )
            for rank, prediction in enumerate(before)
        ]

        return HeadAblationResponse(
            metadata=run.metadata,
            layer=layer,
            head=head,
            tokens=run.tokens,
            before=before,
            after=after,
            deltas=self._prediction_deltas(before, after),
        )

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

    @staticmethod
    def _attention_matrix(token_count: int, head: int) -> list[list[float]]:
        pattern: list[list[float]] = []

        for destination in range(token_count):
            allowed = destination + 1
            row = [0.0] * token_count

            if head % 2 == 0:
                for source in range(allowed):
                    row[source] = round(1.0 / allowed, 6)
            else:
                row[destination] = 0.7
                if destination > 0:
                    row[destination - 1] = 0.3
                else:
                    row[destination] = 1.0

            pattern.append(row)
        return pattern

    @staticmethod
    def _average_entropy(pattern: list[list[float]]) -> float:
        entropies = []
        for row in pattern:
            entropy = -sum(value * math.log(value) for value in row if value > 0)
            entropies.append(entropy)
        return sum(entropies) / len(entropies)

    @staticmethod
    def _previous_token_attention(pattern: list[list[float]]) -> float:
        if len(pattern) <= 1:
            return 0.0
        return sum(pattern[position][position - 1] for position in range(1, len(pattern))) / (
            len(pattern) - 1
        )

    @staticmethod
    def _prediction_deltas(
        before: list[TokenPrediction],
        after: list[TokenPrediction],
    ) -> list[PredictionDelta]:
        deltas: list[PredictionDelta] = []
        for before_prediction, after_prediction in zip(before, after, strict=True):
            deltas.append(
                PredictionDelta(
                    token=before_prediction.token,
                    token_id=before_prediction.token_id,
                    before_logit=before_prediction.logit,
                    after_logit=after_prediction.logit,
                    logit_delta=after_prediction.logit - before_prediction.logit,
                    before_probability=before_prediction.probability,
                    after_probability=after_prediction.probability,
                    probability_delta=after_prediction.probability
                    - before_prediction.probability,
                )
            )
        return deltas
