"""Optional from-scratch spatial-frequency image detector.

Import this module only when the ``neural`` extra is installed. The core
package deliberately remains dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


NEURAL_FORMAT_VERSION = "spatial-frequency-v1"


@dataclass(frozen=True)
class NeuralConfig:
    spatial_widths: tuple[int, ...] = (24, 48, 96)
    frequency_widths: tuple[int, ...] = (16, 32, 64)
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if not self.spatial_widths or not self.frequency_widths:
            raise ValueError("both neural branches require at least one layer")
        if any(width <= 0 for width in self.spatial_widths + self.frequency_widths):
            raise ValueError("branch widths must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")


@dataclass(frozen=True)
class NeuralEvidence:
    logits: Tensor
    spatial_contribution: Tensor
    frequency_contribution: Tensor


class SpatialFrequencyDetector(nn.Module):
    """Compact CNN whose final logit decomposes into two forensic branches."""

    def __init__(self, config: NeuralConfig = NeuralConfig()) -> None:
        super().__init__()
        self.config = config
        self.spatial = _branch(3, config.spatial_widths)
        self.frequency = _branch(1, config.frequency_widths)
        feature_count = config.spatial_widths[-1] + config.frequency_widths[-1]
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(feature_count, 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_evidence(images).logits

    def forward_evidence(self, images: Tensor) -> NeuralEvidence:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if images.shape[-2] < 8 or images.shape[-1] < 8:
            raise ValueError("images must be at least 8 by 8 pixels")
        spatial = self.spatial(images).flatten(1)
        frequency = self.frequency(_log_spectrum(images)).flatten(1)
        features = self.dropout(torch.cat((spatial, frequency), dim=1))
        spatial_count = self.config.spatial_widths[-1]
        weights = self.classifier.weight[0]
        half_bias = self.classifier.bias[0] / 2.0
        spatial_contribution = features[:, :spatial_count] @ weights[:spatial_count] + half_bias
        frequency_contribution = features[:, spatial_count:] @ weights[spatial_count:] + half_bias
        logits = spatial_contribution + frequency_contribution
        return NeuralEvidence(logits, spatial_contribution, frequency_contribution)


def create_scratch_detector(config: NeuralConfig = NeuralConfig(), *, seed: int) -> SpatialFrequencyDetector:
    """Initialize weights deterministically without loading external weights."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SpatialFrequencyDetector(config)


def save_neural_checkpoint(
    path: Path,
    model: SpatialFrequencyDetector,
    *,
    metadata: Mapping[str, str | int | float | bool],
) -> None:
    """Save weights plus auditable primitive metadata; never save executable objects."""

    payload: dict[str, Any] = {
        "format": NEURAL_FORMAT_VERSION,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_neural_checkpoint(path: Path) -> tuple[SpatialFrequencyDetector, dict[str, object]]:
    """Load a versioned weights-only checkpoint onto CPU."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != NEURAL_FORMAT_VERSION:
        raise ValueError("unsupported neural checkpoint format")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint config is missing")
    config = NeuralConfig(
        spatial_widths=tuple(int(value) for value in raw_config["spatial_widths"]),
        frequency_widths=tuple(int(value) for value in raw_config["frequency_widths"]),
        dropout=float(raw_config["dropout"]),
    )
    model = SpatialFrequencyDetector(config)
    model.load_state_dict(payload["state_dict"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return model, metadata


def _branch(input_channels: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = input_channels
    for width in widths:
        layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.GELU()))
        channels = width
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)


def _log_spectrum(images: Tensor) -> Tensor:
    gray = images.mean(dim=1, keepdim=True)
    spectrum = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))))
    mean = spectrum.mean(dim=(-2, -1), keepdim=True)
    scale = spectrum.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (spectrum - mean) / scale
