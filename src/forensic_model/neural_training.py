"""Reproducible optimization and evaluation for the optional neural model."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forensic_model.calibration import PlattCalibrator
from forensic_model.metrics import BinaryMetrics, binary_metrics
from forensic_model.neural import NeuralConfig, SpatialFrequencyDetector, create_scratch_detector


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260818
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    cpu_threads: int = 8

    def __post_init__(self) -> None:
        if self.seed < 0 or self.epochs <= 0 or self.batch_size <= 0 or self.cpu_threads <= 0:
            raise ValueError("seed must be non-negative and counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")


@dataclass(frozen=True)
class TrainingEpoch:
    epoch: int
    train_loss: float
    validation_auroc: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainedNeuralDetector:
    model: SpatialFrequencyDetector
    calibrator: PlattCalibrator
    threshold: float
    history: tuple[TrainingEpoch, ...]
    training_config: TrainingConfig
    frequency_weight: float = 1.0

    def metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "initialization": "scratch",
            "seed": self.training_config.seed,
            "epochs": self.training_config.epochs,
            "threshold": self.threshold,
            "calibration_slope": self.calibrator.slope,
            "calibration_intercept": self.calibrator.intercept,
            "frequency_weight": self.frequency_weight,
        }


def train_neural_detector(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    training_config: TrainingConfig = TrainingConfig(),
    model_config: NeuralConfig = NeuralConfig(),
) -> TrainedNeuralDetector:
    """Train from scratch and fit calibration and threshold on validation only."""

    _seed_everything(training_config)
    model = create_scratch_detector(model_config, seed=training_config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
    loss_function = nn.BCEWithLogitsLoss()
    train_loader = _loader(train_dataset, training_config, shuffle=True)
    validation_loader = _loader(validation_dataset, training_config, shuffle=False)
    history = []
    for epoch in range(1, training_config.epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            sample_count += len(images)
        scheduler.step()
        validation_scores, validation_labels, _ = score_neural_detector(model, validation_loader)
        validation_probabilities = torch.sigmoid(torch.tensor(validation_scores)).tolist()
        history.append(
            TrainingEpoch(
                epoch=epoch,
                train_loss=total_loss / sample_count,
                validation_auroc=binary_metrics(validation_labels, validation_probabilities).auroc,
                elapsed_seconds=time.monotonic() - started,
            )
        )

    validation_scores, validation_labels, _ = score_neural_detector(model, validation_loader)
    calibrator = PlattCalibrator.fit(validation_scores, validation_labels)
    probabilities = [calibrator.transform(score) for score in validation_scores]
    threshold = balanced_accuracy_threshold(validation_labels, probabilities)
    return TrainedNeuralDetector(model, calibrator, threshold, tuple(history), training_config)


@torch.inference_mode()
def score_neural_detector(
    model: SpatialFrequencyDetector,
    loader: DataLoader,
    *,
    frequency_weight: float = 1.0,
) -> tuple[list[float], list[int], list[str]]:
    if not 0.0 <= frequency_weight <= 1.0:
        raise ValueError("frequency_weight must be within [0, 1]")
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    for images, targets, identities in loader:
        evidence = model.forward_evidence(images)
        scores.extend((evidence.spatial_contribution + frequency_weight * evidence.frequency_contribution).tolist())
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    return scores, labels, groups


def evaluate_neural_detector(
    detector: TrainedNeuralDetector,
    dataset: Dataset,
) -> BinaryMetrics:
    loader = _loader(dataset, detector.training_config, shuffle=False)
    scores, labels, _ = score_neural_detector(
        detector.model, loader, frequency_weight=detector.frequency_weight
    )
    probabilities = [detector.calibrator.transform(score) for score in scores]
    return binary_metrics(labels, probabilities, threshold=detector.threshold)


def recalibrate_neural_detector(
    detector: TrainedNeuralDetector,
    validation_dataset: Dataset,
    *,
    frequency_weight: float,
) -> TrainedNeuralDetector:
    """Refit confidence and threshold after a validation-selected branch blend."""

    loader = _loader(validation_dataset, detector.training_config, shuffle=False)
    scores, labels, _ = score_neural_detector(
        detector.model, loader, frequency_weight=frequency_weight
    )
    calibrator = PlattCalibrator.fit(scores, labels)
    probabilities = [calibrator.transform(score) for score in scores]
    return TrainedNeuralDetector(
        detector.model,
        calibrator,
        balanced_accuracy_threshold(labels, probabilities),
        detector.history,
        detector.training_config,
        frequency_weight,
    )


def balanced_accuracy_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select a validation threshold in O(n log n), handling tied scores together."""

    if len(labels) != len(probabilities) or set(labels) != {0, 1}:
        raise ValueError("threshold selection requires aligned binary labels")
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positive = 0
    false_positive = 0
    best_score = 0.5
    best_threshold = 0.5
    ordered = sorted(zip(probabilities, labels), reverse=True)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positive += sum(label for _, label in ordered[index:end])
        false_positive += (end - index) - sum(label for _, label in ordered[index:end])
        score = (true_positive / positives + (negatives - false_positive) / negatives) / 2.0
        if score > best_score:
            best_score = score
            best_threshold = ordered[index][0]
        index = end
    return min(max(best_threshold, 1e-9), 1.0 - 1e-9)


def history_as_dicts(history: Sequence[TrainingEpoch]) -> list[dict[str, int | float]]:
    return [asdict(epoch) for epoch in history]


def _loader(dataset: Dataset, config: TrainingConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _seed_everything(config: TrainingConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config.cpu_threads)
