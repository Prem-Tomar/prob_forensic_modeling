"""Frozen image-detector aggregation baseline for temporal comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from forensic_model.calibration import PlattCalibrator
from forensic_model.neural import SpatialFrequencyDetector
from forensic_model.neural_inference import CalibratedNeuralImageDetector
from forensic_model.neural_training import balanced_accuracy_threshold


@dataclass(frozen=True)
class FrameAggregationResult:
    probability_synthetic: float
    frame_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class FrozenFrameAggregator:
    model: SpatialFrequencyDetector
    calibrator: PlattCalibrator
    threshold: float
    frequency_weight: float = 1.0

    @classmethod
    def load(cls, checkpoint: Path) -> "FrozenFrameAggregator":
        detector = CalibratedNeuralImageDetector.load(checkpoint, abstain_margin=0.0)
        return cls(
            detector.model,
            detector.calibrator,
            detector.threshold,
            detector.frequency_weight,
        )

    @torch.inference_mode()
    def predict(self, frames: Tensor) -> FrameAggregationResult:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [time, 3, height, width]")
        evidence = self.model.forward_evidence(frames)
        scores = evidence.spatial_contribution + self.frequency_weight * evidence.frequency_contribution
        probabilities = tuple(self.calibrator.transform(float(score)) for score in scores)
        if not probabilities:
            raise ValueError("at least one frame is required")
        return FrameAggregationResult(sum(probabilities) / len(probabilities), probabilities)


@dataclass(frozen=True)
class ValidatedFrameAggregator:
    frame_aggregator: FrozenFrameAggregator
    clip_calibrator: PlattCalibrator
    threshold: float

    def transform(self, probability: float) -> float:
        return self.clip_calibrator.transform(probability)


@torch.inference_mode()
def score_frame_aggregation(
    aggregator: FrozenFrameAggregator,
    dataset: Dataset,
    *,
    batch_size: int = 8,
) -> tuple[list[float], list[int], list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    probabilities = []
    labels = []
    groups = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    aggregator.model.eval()
    for clips, targets, identities in loader:
        batch, time = clips.shape[:2]
        evidence = aggregator.model.forward_evidence(clips.flatten(0, 1))
        scores = evidence.spatial_contribution + aggregator.frequency_weight * evidence.frequency_contribution
        frame_probabilities = torch.tensor(
            [aggregator.calibrator.transform(float(score)) for score in scores]
        ).reshape(batch, time)
        probabilities.extend(frame_probabilities.mean(dim=1).tolist())
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    return probabilities, labels, groups


def fit_clip_aggregation(
    aggregator: FrozenFrameAggregator,
    validation_dataset: Dataset,
    *,
    batch_size: int = 8,
) -> ValidatedFrameAggregator:
    """Calibrate clip means and choose their threshold using validation only."""

    scores, labels, _ = score_frame_aggregation(
        aggregator,
        validation_dataset,
        batch_size=batch_size,
    )
    calibrator = PlattCalibrator.fit(scores, labels)
    probabilities = [calibrator.transform(score) for score in scores]
    return ValidatedFrameAggregator(
        aggregator,
        calibrator,
        balanced_accuracy_threshold(labels, probabilities),
    )


def score_validated_frame_aggregation(
    aggregator: ValidatedFrameAggregator,
    dataset: Dataset,
    *,
    batch_size: int = 8,
) -> tuple[list[float], list[int], list[str]]:
    scores, labels, groups = score_frame_aggregation(
        aggregator.frame_aggregator,
        dataset,
        batch_size=batch_size,
    )
    return [aggregator.transform(score) for score in scores], labels, groups
