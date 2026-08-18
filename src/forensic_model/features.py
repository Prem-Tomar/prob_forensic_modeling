"""Versioned, interpretable image features for the baseline model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from forensic_model.image import RGBImage


FEATURE_VERSION = "forensic-summary-v1"
FEATURE_NAMES = (
    "red_mean",
    "green_mean",
    "blue_mean",
    "luma_std",
    "saturation_mean",
    "horizontal_residual",
    "vertical_residual",
    "laplacian_energy",
    "checkerboard_energy",
    "clipped_fraction",
)


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    version: str = FEATURE_VERSION

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


def extract_features(image: RGBImage) -> FeatureVector:
    """Extract bounded summaries spanning color, noise, and frequency cues."""

    channels = tuple(zip(*image.pixels))
    means = tuple(sum(channel) / len(channel) for channel in channels)
    luma = tuple(0.2126 * red + 0.7152 * green + 0.0722 * blue for red, green, blue in image.pixels)
    luma_mean = sum(luma) / len(luma)
    luma_std = math.sqrt(sum((value - luma_mean) ** 2 for value in luma) / len(luma))
    saturation = sum(max(pixel) - min(pixel) for pixel in image.pixels) / len(image.pixels)

    horizontal = _mean_abs_difference(image, luma, 1, 0)
    vertical = _mean_abs_difference(image, luma, 0, 1)
    laplacian = _laplacian_energy(image, luma)
    checkerboard = abs(
        sum((1.0 if (index % image.width + index // image.width) % 2 == 0 else -1.0) * value for index, value in enumerate(luma))
        / len(luma)
    )
    clipped = sum(channel <= 1.0 / 255.0 or channel >= 254.0 / 255.0 for pixel in image.pixels for channel in pixel) / (3 * len(image.pixels))

    return FeatureVector(
        names=FEATURE_NAMES,
        values=means + (luma_std, saturation, horizontal, vertical, laplacian, checkerboard, clipped),
    )


def _mean_abs_difference(image: RGBImage, luma: tuple[float, ...], dx: int, dy: int) -> float:
    differences = []
    for y in range(image.height - dy):
        for x in range(image.width - dx):
            index = y * image.width + x
            other = (y + dy) * image.width + x + dx
            differences.append(abs(luma[index] - luma[other]))
    return sum(differences) / len(differences)


def _laplacian_energy(image: RGBImage, luma: tuple[float, ...]) -> float:
    if image.width < 3 or image.height < 3:
        return 0.0
    residuals = []
    for y in range(1, image.height - 1):
        for x in range(1, image.width - 1):
            center = y * image.width + x
            value = 4 * luma[center] - luma[center - 1] - luma[center + 1] - luma[center - image.width] - luma[center + image.width]
            residuals.append(abs(value))
    return sum(residuals) / len(residuals)
