"""Deterministic scratch training and calibration for neural video models."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forensic_model.calibration import PlattCalibrator
from forensic_model.metrics import BinaryMetrics, binary_metrics
from forensic_model.neural_training import balanced_accuracy_threshold
from forensic_model.neural_video import TemporalConfig, TemporalResidualDetector, create_scratch_temporal_detector


@dataclass(frozen=True)
class TemporalTrainingConfig:
    seed: int = 20260818
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    cpu_threads: int = 8

    def __post_init__(self) -> None:
        counts = (self.epochs, self.batch_size, self.patience, self.cpu_threads)
        if self.seed < 0 or any(value <= 0 for value in counts):
            raise ValueError("seed must be non-negative and counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")


@dataclass(frozen=True)
class TemporalTrainingEpoch:
    epoch: int
    train_loss: float
    validation_auroc: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainedTemporalDetector:
    model: TemporalResidualDetector
    calibrator: PlattCalibrator
    threshold: float
    history: tuple[TemporalTrainingEpoch, ...]
    training_config: TemporalTrainingConfig

    def metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "initialization": "scratch",
            "seed": self.training_config.seed,
            "epochs_completed": len(self.history),
            "threshold": self.threshold,
            "calibration_slope": self.calibrator.slope,
            "calibration_intercept": self.calibrator.intercept,
        }


def train_temporal_detector(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    training_config: TemporalTrainingConfig = TemporalTrainingConfig(),
    model_config: TemporalConfig = TemporalConfig(),
) -> TrainedTemporalDetector:
    """Train from scratch, selecting epochs and calibration on validation only."""

    _seed_everything(training_config)
    model = create_scratch_temporal_detector(model_config, seed=training_config.seed)
    negative_count, positive_count = _label_counts(train_dataset)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
    train_loader = _loader(train_dataset, training_config, shuffle=True)
    validation_loader = _loader(validation_dataset, training_config, shuffle=False)
    history = []
    best_auroc = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, training_config.epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        sample_count = 0
        for frames, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(frames), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(frames)
            sample_count += len(frames)
        scheduler.step()
        scores, labels, _ = score_temporal_detector(model, validation_loader)
        probabilities = torch.sigmoid(torch.tensor(scores)).tolist()
        validation_auroc = binary_metrics(labels, probabilities).auroc
        history.append(
            TemporalTrainingEpoch(
                epoch,
                total_loss / sample_count,
                validation_auroc,
                time.monotonic() - started,
            )
        )
        if validation_auroc > best_auroc:
            best_auroc = validation_auroc
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break
    if best_state is None:
        raise RuntimeError("temporal training produced no checkpoint")
    model.load_state_dict(best_state)
    scores, labels, _ = score_temporal_detector(model, validation_loader)
    calibrator = PlattCalibrator.fit(scores, labels)
    probabilities = [calibrator.transform(score) for score in scores]
    threshold = balanced_accuracy_threshold(labels, probabilities)
    return TrainedTemporalDetector(model, calibrator, threshold, tuple(history), training_config)


@torch.inference_mode()
def score_temporal_detector(
    model: TemporalResidualDetector,
    loader: DataLoader,
) -> tuple[list[float], list[int], list[str]]:
    model.eval()
    scores = []
    labels = []
    groups = []
    for frames, targets, identities in loader:
        scores.extend(model(frames).tolist())
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    return scores, labels, groups


def evaluate_temporal_detector(
    detector: TrainedTemporalDetector,
    dataset: Dataset,
) -> BinaryMetrics:
    loader = _loader(dataset, detector.training_config, shuffle=False)
    scores, labels, _ = score_temporal_detector(detector.model, loader)
    probabilities = [detector.calibrator.transform(score) for score in scores]
    return binary_metrics(labels, probabilities, threshold=detector.threshold)


def _label_counts(dataset: Dataset) -> tuple[int, int]:
    examples = getattr(dataset, "examples", None)
    if examples is None:
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
    else:
        labels = [int(example.label) for example in examples]
    negative_count = labels.count(0)
    positive_count = labels.count(1)
    if not negative_count or not positive_count:
        raise ValueError("temporal training requires both labels")
    return negative_count, positive_count


def _loader(dataset: Dataset, config: TemporalTrainingConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _seed_everything(config: TemporalTrainingConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config.cpu_threads)
