"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_MODELS = ("gpt2-small", "distilgpt2", "gelu-1l", "attn-only-1l")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the API and model service."""

    default_model: str = os.getenv("TCV_DEFAULT_MODEL", "gpt2-small")
    available_models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODELS)
    device: str | None = os.getenv("TCV_DEVICE")


settings = Settings()
