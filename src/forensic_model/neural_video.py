"""From-scratch neural video detector with decomposed temporal evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


TEMPORAL_FORMAT_VERSION = "temporal-residual-v1"


@dataclass(frozen=True)
class TemporalConfig:
    frame_widths: tuple[int, ...] = (12, 24, 48)
    temporal_hidden: int = 32
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if not self.frame_widths or any(width <= 0 for width in self.frame_widths):
            raise ValueError("frame widths must be positive")
        if self.temporal_hidden <= 0:
            raise ValueError("temporal_hidden must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")


@dataclass(frozen=True)
class TemporalEvidence:
    logits: Tensor
    frame_contribution: Tensor
    temporal_contribution: Tensor
    frame_logits: Tensor


class TemporalResidualDetector(nn.Module):
    """Separate appearance evidence from changes between adjacent frames."""

    def __init__(self, config: TemporalConfig = TemporalConfig()) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = _frame_encoder(config.frame_widths)
        embedding_width = config.frame_widths[-1]
        self.dropout = nn.Dropout(config.dropout)
        self.frame_classifier = nn.Linear(embedding_width, 1)
        self.temporal_encoder = nn.GRU(embedding_width, config.temporal_hidden, batch_first=True)
        self.temporal_classifier = nn.Linear(config.temporal_hidden, 1)

    def forward(self, frames: Tensor) -> Tensor:
        return self.forward_evidence(frames).logits

    def forward_evidence(self, frames: Tensor) -> TemporalEvidence:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch, time, 3, height, width]")
        if frames.shape[1] < 3:
            raise ValueError("at least three frames are required")
        if frames.shape[-2] < 8 or frames.shape[-1] < 8:
            raise ValueError("frames must be at least 8 by 8 pixels")

        batch, time = frames.shape[:2]
        embeddings = self.frame_encoder(frames.flatten(0, 1)).flatten(1).reshape(batch, time, -1)
        frame_logits = self.frame_classifier(self.dropout(embeddings)).squeeze(-1)
        frame_contribution = frame_logits.mean(dim=1)
        residuals = embeddings[:, 1:] - embeddings[:, :-1]
        temporal_states, _ = self.temporal_encoder(residuals)
        temporal_contribution = self.temporal_classifier(self.dropout(temporal_states[:, -1])).squeeze(-1)
        logits = frame_contribution + temporal_contribution
        return TemporalEvidence(logits, frame_contribution, temporal_contribution, frame_logits)


def create_scratch_temporal_detector(
    config: TemporalConfig = TemporalConfig(),
    *,
    seed: int,
) -> TemporalResidualDetector:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return TemporalResidualDetector(config)


def save_temporal_checkpoint(
    path: Path,
    model: TemporalResidualDetector,
    *,
    metadata: Mapping[str, str | int | float | bool],
) -> None:
    payload: dict[str, Any] = {
        "format": TEMPORAL_FORMAT_VERSION,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_temporal_checkpoint(path: Path) -> tuple[TemporalResidualDetector, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != TEMPORAL_FORMAT_VERSION:
        raise ValueError("unsupported temporal checkpoint format")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("temporal checkpoint config is missing")
    config = TemporalConfig(
        frame_widths=tuple(int(value) for value in raw_config["frame_widths"]),
        temporal_hidden=int(raw_config["temporal_hidden"]),
        dropout=float(raw_config["dropout"]),
    )
    model = TemporalResidualDetector(config)
    model.load_state_dict(payload["state_dict"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("temporal checkpoint metadata must be a mapping")
    return model, metadata


def _frame_encoder(widths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = 3
    for width in widths:
        layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.GELU()))
        channels = width
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)
