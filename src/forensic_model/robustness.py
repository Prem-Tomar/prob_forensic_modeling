"""Unseen-family slices and deterministic image post-processing stresses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from forensic_model.detector import ImageDetector
from forensic_model.image import RGBImage
from forensic_model.metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class EvaluationSample:
    image: RGBImage
    label: int
    content_group: str
    generator_family: str = ""


@dataclass(frozen=True)
class Transformation:
    name: str
    apply: Callable[[RGBImage], RGBImage]


def evaluate_samples(detector: ImageDetector, samples: Sequence[EvaluationSample]) -> BinaryMetrics:
    probabilities = [detector.analyze_image(sample.image).probability_synthetic for sample in samples]
    return binary_metrics([sample.label for sample in samples], probabilities, threshold=detector.policy.threshold)


def evaluate_unseen_generators(
    detector: ImageDetector,
    samples: Sequence[EvaluationSample],
    *,
    trained_generator_families: Iterable[str],
) -> dict[str, BinaryMetrics]:
    trained = {family.casefold() for family in trained_generator_families}
    families = sorted({sample.generator_family for sample in samples if sample.label == 1})
    overlap = [family for family in families if family.casefold() in trained]
    if overlap:
        raise ValueError(f"evaluation families were seen during training: {', '.join(overlap)}")
    real = [sample for sample in samples if sample.label == 0]
    return {
        family: evaluate_samples(detector, real + [sample for sample in samples if sample.generator_family == family])
        for family in families
    }


def evaluate_postprocessing(
    detector: ImageDetector,
    samples: Sequence[EvaluationSample],
    transformations: Sequence[Transformation] | None = None,
) -> dict[str, BinaryMetrics]:
    results = {"clean": evaluate_samples(detector, samples)}
    for transformation in transformations or default_transformations():
        transformed = [
            EvaluationSample(transformation.apply(sample.image), sample.label, sample.content_group, sample.generator_family)
            for sample in samples
        ]
        results[transformation.name] = evaluate_samples(detector, transformed)
    return results


def default_transformations() -> tuple[Transformation, ...]:
    return (
        Transformation("gamma_0.8", lambda image: gamma(image, 0.8)),
        Transformation("gamma_1.2", lambda image: gamma(image, 1.2)),
        Transformation("box_blur_1", lambda image: box_blur(image, 1)),
        Transformation("noise_0.02", lambda image: add_noise(image, 0.02, seed=17)),
        Transformation("quantize_32", lambda image: quantize(image, 32)),
        Transformation("center_crop_0.75", lambda image: center_crop_rescale(image, 0.75)),
    )


def gamma(image: RGBImage, exponent: float) -> RGBImage:
    if exponent <= 0.0 or not math.isfinite(exponent):
        raise ValueError("gamma exponent must be positive and finite")
    return _map_pixels(image, lambda channel: channel**exponent)


def quantize(image: RGBImage, levels: int) -> RGBImage:
    if levels < 2 or levels > 256:
        raise ValueError("quantization levels must be between 2 and 256")
    maximum = levels - 1
    return _map_pixels(image, lambda channel: round(channel * maximum) / maximum)


def add_noise(image: RGBImage, amplitude: float, *, seed: int) -> RGBImage:
    if amplitude < 0.0 or amplitude > 1.0:
        raise ValueError("noise amplitude must be within [0, 1]")
    randomizer = random.Random(seed)
    pixels = tuple(
        tuple(_clip(channel + randomizer.uniform(-amplitude, amplitude)) for channel in pixel)
        for pixel in image.pixels
    )
    return RGBImage(image.width, image.height, pixels)  # type: ignore[arg-type]


def box_blur(image: RGBImage, radius: int) -> RGBImage:
    if radius < 1:
        raise ValueError("blur radius must be positive")
    pixels = []
    for y in range(image.height):
        for x in range(image.width):
            neighbors = [
                image.pixel(nx, ny)
                for ny in range(max(0, y - radius), min(image.height, y + radius + 1))
                for nx in range(max(0, x - radius), min(image.width, x + radius + 1))
            ]
            pixels.append(tuple(sum(pixel[channel] for pixel in neighbors) / len(neighbors) for channel in range(3)))
    return RGBImage(image.width, image.height, tuple(pixels))  # type: ignore[arg-type]


def center_crop_rescale(image: RGBImage, fraction: float) -> RGBImage:
    if not 0.25 <= fraction <= 1.0:
        raise ValueError("crop fraction must be within [0.25, 1]")
    crop_width = max(2, round(image.width * fraction))
    crop_height = max(2, round(image.height * fraction))
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    pixels = []
    for y in range(image.height):
        source_y = top + min(crop_height - 1, y * crop_height // image.height)
        for x in range(image.width):
            source_x = left + min(crop_width - 1, x * crop_width // image.width)
            pixels.append(image.pixel(source_x, source_y))
    return RGBImage(image.width, image.height, tuple(pixels))


def _map_pixels(image: RGBImage, transform: Callable[[float], float]) -> RGBImage:
    pixels = tuple(tuple(_clip(transform(channel)) for channel in pixel) for pixel in image.pixels)
    return RGBImage(image.width, image.height, pixels)  # type: ignore[arg-type]


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)
