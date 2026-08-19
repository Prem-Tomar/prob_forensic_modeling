"""Reusable calibrated inference for the optional neural image detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from forensic_model.calibration import PlattCalibrator
from forensic_model.neural import SpatialFrequencyDetector, load_neural_checkpoint
from forensic_model.neural_data import evaluation_transform


@dataclass(frozen=True)
class NeuralImageResult:
    """Calibrated decision with the deployed and raw branch evidence."""

    decision: str
    probability_synthetic: float
    decision_confidence: float
    raw_score: float
    threshold: float
    abstained: bool
    spatial_contribution: float
    raw_frequency_contribution: float
    deployed_frequency_contribution: float


@dataclass(frozen=True)
class CalibratedNeuralImageDetector:
    """Load and apply a from-scratch checkpoint without retraining."""

    model: SpatialFrequencyDetector
    calibrator: PlattCalibrator
    threshold: float
    frequency_weight: float
    abstain_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("checkpoint threshold must be between zero and one")
        if not 0.0 <= self.frequency_weight <= 1.0:
            raise ValueError("checkpoint frequency weight must be within [0, 1]")
        if not 0.0 <= self.abstain_margin < min(self.threshold, 1.0 - self.threshold):
            raise ValueError("abstain margin must fit on both sides of the threshold")
        self.model.eval()

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        *,
        abstain_margin: float = 0.05,
    ) -> "CalibratedNeuralImageDetector":
        model, metadata = load_neural_checkpoint(checkpoint)
        required = ("calibration_slope", "calibration_intercept", "threshold")
        if any(name not in metadata for name in required):
            raise ValueError("neural checkpoint lacks calibrated inference metadata")
        return cls(
            model,
            PlattCalibrator(
                float(metadata["calibration_slope"]),
                float(metadata["calibration_intercept"]),
            ),
            float(metadata["threshold"]),
            float(metadata.get("frequency_weight", 1.0)),
            abstain_margin,
        )

    def predict_file(self, path: Path, *, image_size: int = 32) -> NeuralImageResult:
        """Decode one local image and return calibrated explainable evidence."""

        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor = evaluation_transform(image_size)(image)
        return self.predict_tensors(tensor.unsqueeze(0))[0]

    @torch.inference_mode()
    def predict_tensors(self, images: Tensor) -> tuple[NeuralImageResult, ...]:
        """Predict a non-empty RGB tensor batch shaped [batch, 3, height, width]."""

        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
            raise ValueError("images must have shape [non-empty batch, 3, height, width]")
        evidence = self.model.forward_evidence(images)
        deployed_frequency = self.frequency_weight * evidence.frequency_contribution
        scores = evidence.spatial_contribution + deployed_frequency
        return tuple(
            self._result(
                float(score),
                float(spatial),
                float(raw_frequency),
                float(weighted_frequency),
            )
            for score, spatial, raw_frequency, weighted_frequency in zip(
                scores,
                evidence.spatial_contribution,
                evidence.frequency_contribution,
                deployed_frequency,
            )
        )

    def _result(
        self,
        score: float,
        spatial: float,
        raw_frequency: float,
        deployed_frequency: float,
    ) -> NeuralImageResult:
        probability = self.calibrator.transform(score)
        abstained = abs(probability - self.threshold) <= self.abstain_margin
        if abstained:
            decision = "abstain"
        elif probability > self.threshold:
            decision = "synthetic"
        else:
            decision = "camera_or_human"
        return NeuralImageResult(
            decision,
            probability,
            max(probability, 1.0 - probability),
            score,
            self.threshold,
            abstained,
            spatial,
            raw_frequency,
            deployed_frequency,
        )
