"""Vectorized real-data training for the dependency-free Phase 1 image model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy
from forensic_model.detector import ImageDetector
from forensic_model.features import FEATURE_NAMES, FEATURE_VERSION, FeatureVector
from forensic_model.model import LogisticModel
from forensic_model.neural_training import balanced_accuracy_threshold


@dataclass(frozen=True)
class FeatureTrainingConfig:
    learning_rate: float = 0.15
    epochs: int = 800
    l2: float = 0.01
    batch_size: int = 512
    cpu_threads: int = 8
    seed: int = 20260818

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.epochs <= 0 or self.l2 < 0.0:
            raise ValueError("invalid feature optimizer configuration")
        if self.batch_size <= 0 or self.cpu_threads <= 0 or self.seed < 0:
            raise ValueError("feature counts must be positive and seed must be non-negative")


@dataclass(frozen=True)
class TrainedFeatureDetector:
    detector: ImageDetector
    final_training_loss: float
    training_config: FeatureTrainingConfig


def extract_summary_feature_matrix(images: Tensor) -> Tensor:
    """Compute the exact ten Phase 1 summaries for an RGB tensor batch."""

    if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
        raise ValueError("images must have shape [non-empty batch, 3, height, width]")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("images must be at least 2 by 2 pixels")
    values = images.to(dtype=torch.float64)
    means = values.mean(dim=(-2, -1))
    luma = 0.2126 * values[:, 0] + 0.7152 * values[:, 1] + 0.0722 * values[:, 2]
    luma_mean = luma.mean(dim=(-2, -1), keepdim=True)
    luma_std = ((luma - luma_mean).square().mean(dim=(-2, -1))).sqrt()
    saturation = (values.max(dim=1).values - values.min(dim=1).values).mean(dim=(-2, -1))
    horizontal = (luma[:, :, 1:] - luma[:, :, :-1]).abs().mean(dim=(-2, -1))
    vertical = (luma[:, 1:, :] - luma[:, :-1, :]).abs().mean(dim=(-2, -1))
    if images.shape[-2] < 3 or images.shape[-1] < 3:
        laplacian = torch.zeros(len(images), dtype=torch.float64, device=values.device)
    else:
        center = luma[:, 1:-1, 1:-1]
        residual = (
            4.0 * center
            - luma[:, 1:-1, :-2]
            - luma[:, 1:-1, 2:]
            - luma[:, :-2, 1:-1]
            - luma[:, 2:, 1:-1]
        )
        laplacian = residual.abs().mean(dim=(-2, -1))
    rows = torch.arange(images.shape[-2], device=values.device).reshape(-1, 1)
    columns = torch.arange(images.shape[-1], device=values.device).reshape(1, -1)
    checkerboard_sign = ((rows + columns) % 2 == 0).to(torch.float64) * 2.0 - 1.0
    checkerboard = (luma * checkerboard_sign).mean(dim=(-2, -1)).abs()
    clipped = ((values <= 1.0 / 255.0) | (values >= 254.0 / 255.0)).to(torch.float64).mean(dim=(1, 2, 3))
    return torch.cat(
        (
            means,
            luma_std[:, None],
            saturation[:, None],
            horizontal[:, None],
            vertical[:, None],
            laplacian[:, None],
            checkerboard[:, None],
            clipped[:, None],
        ),
        dim=1,
    )


def extract_dataset_features(
    dataset: Dataset,
    *,
    batch_size: int,
) -> tuple[Tensor, list[int], list[str]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    matrices = []
    labels: list[int] = []
    groups: list[str] = []
    for images, targets, identities in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        matrices.append(extract_summary_feature_matrix(images))
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    if not matrices:
        raise ValueError("feature dataset must not be empty")
    return torch.cat(matrices), labels, groups


def fit_feature_detector(
    training_features: Tensor,
    training_labels: Sequence[int],
    validation_features: Tensor,
    validation_labels: Sequence[int],
    *,
    config: FeatureTrainingConfig = FeatureTrainingConfig(),
) -> TrainedFeatureDetector:
    """Fit the Phase 1 artifact and validation-only calibration deterministically."""

    if training_features.ndim != 2 or training_features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("training features do not match the Phase 1 schema")
    if validation_features.ndim != 2 or validation_features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("validation features do not match the Phase 1 schema")
    if len(training_features) != len(training_labels) or set(training_labels) != {0, 1}:
        raise ValueError("training features require aligned binary labels")
    if len(validation_features) != len(validation_labels) or set(validation_labels) != {0, 1}:
        raise ValueError("validation features require aligned binary labels")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        matrix = training_features.to(dtype=torch.float64, device="cpu")
        targets = torch.tensor(training_labels, dtype=torch.float64)
        means = matrix.mean(dim=0)
        scales = ((matrix - means).square().mean(dim=0)).sqrt().clamp_min(1e-9)
        standardized = (matrix - means) / scales
        weights = torch.zeros(len(FEATURE_NAMES), dtype=torch.float64)
        bias = torch.tensor(0.0, dtype=torch.float64)
        for _ in range(config.epochs):
            probabilities = torch.sigmoid(standardized @ weights + bias)
            errors = probabilities - targets
            weights -= config.learning_rate * (standardized.T @ errors / len(matrix) + config.l2 * weights)
            bias -= config.learning_rate * errors.mean()
        final_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(standardized @ weights + bias, targets)
            + config.l2 * weights.square().sum() / 2.0
        )
    finally:
        torch.set_num_threads(previous_threads)
    model = LogisticModel(
        FEATURE_NAMES,
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        tuple(float(value) for value in weights),
        float(bias),
        FEATURE_VERSION,
    )
    validation_scores = score_feature_model(model, validation_features)
    calibrator = PlattCalibrator.fit(validation_scores, validation_labels)
    probabilities = [calibrator.transform(score) for score in validation_scores]
    threshold = balanced_accuracy_threshold(validation_labels, probabilities)
    detector = ImageDetector(model, calibrator, DecisionPolicy(threshold=threshold, abstain_margin=0.0))
    return TrainedFeatureDetector(detector, final_loss, config)


def score_feature_model(model: LogisticModel, features: Tensor) -> list[float]:
    means = torch.tensor(model.means, dtype=torch.float64, device=features.device)
    scales = torch.tensor(model.scales, dtype=torch.float64, device=features.device)
    weights = torch.tensor(model.weights, dtype=torch.float64, device=features.device)
    standardized = ((features.to(torch.float64) - means) / scales).clamp(
        -model.standardized_clip,
        model.standardized_clip,
    )
    return (standardized @ weights + model.bias).tolist()


def feature_vectors(matrix: Tensor) -> tuple[FeatureVector, ...]:
    """Convert a tensor matrix to the public dependency-free feature type."""

    return tuple(FeatureVector(FEATURE_NAMES, tuple(float(value) for value in row)) for row in matrix)
