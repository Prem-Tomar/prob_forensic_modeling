"""Reusable calibrated inference for the optional temporal video detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from forensic_model.calibration import PlattCalibrator
from forensic_model.neural_video import TemporalResidualDetector, load_temporal_checkpoint


@dataclass(frozen=True)
class TemporalResult:
    decision: str
    probability_synthetic: float
    decision_confidence: float
    raw_score: float
    threshold: float
    abstained: bool
    frame_contribution: float
    temporal_contribution: float
    frame_logits: tuple[float, ...]


@dataclass(frozen=True)
class CalibratedTemporalDetector:
    model: TemporalResidualDetector
    calibrator: PlattCalibrator
    threshold: float
    abstain_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("checkpoint threshold must be between zero and one")
        if not 0.0 <= self.abstain_margin < min(self.threshold, 1.0 - self.threshold):
            raise ValueError("abstain margin must fit on both sides of the threshold")
        self.model.eval()

    @classmethod
    def load(cls, checkpoint: Path, *, abstain_margin: float = 0.05) -> "CalibratedTemporalDetector":
        model, metadata = load_temporal_checkpoint(checkpoint)
        required = ("calibration_slope", "calibration_intercept", "threshold")
        if any(name not in metadata for name in required):
            raise ValueError("temporal checkpoint lacks calibrated inference metadata")
        return cls(
            model,
            PlattCalibrator(
                float(metadata["calibration_slope"]),
                float(metadata["calibration_intercept"]),
            ),
            float(metadata["threshold"]),
            abstain_margin,
        )

    @torch.inference_mode()
    def predict_tensors(self, clips: Tensor) -> tuple[TemporalResult, ...]:
        """Predict a non-empty batch shaped [batch, time, 3, height, width]."""

        if clips.ndim != 5 or clips.shape[0] == 0 or clips.shape[2] != 3:
            raise ValueError("clips must have shape [non-empty batch, time, 3, height, width]")
        evidence = self.model.forward_evidence(clips)
        return tuple(
            self._result(
                float(score),
                float(frame),
                float(temporal),
                tuple(float(value) for value in frame_logits),
            )
            for score, frame, temporal, frame_logits in zip(
                evidence.logits,
                evidence.frame_contribution,
                evidence.temporal_contribution,
                evidence.frame_logits,
            )
        )

    def _result(
        self,
        score: float,
        frame: float,
        temporal: float,
        frame_logits: tuple[float, ...],
    ) -> TemporalResult:
        probability = self.calibrator.transform(score)
        abstained = abs(probability - self.threshold) <= self.abstain_margin
        if abstained:
            decision = "abstain"
        elif probability > self.threshold:
            decision = "synthetic"
        else:
            decision = "camera_or_human"
        return TemporalResult(
            decision,
            probability,
            max(probability, 1.0 - probability),
            score,
            self.threshold,
            abstained,
            frame,
            temporal,
            frame_logits,
        )
