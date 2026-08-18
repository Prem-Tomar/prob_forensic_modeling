"""Explainable confidence and abstention for model predictions."""

from __future__ import annotations

from dataclasses import dataclass

from forensic_model.calibration import PlattCalibrator
from forensic_model.model import Prediction


@dataclass(frozen=True)
class DecisionPolicy:
    threshold: float = 0.5
    abstain_margin: float = 0.1
    reason_count: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if not 0.0 <= self.abstain_margin < min(self.threshold, 1.0 - self.threshold):
            raise ValueError("abstain_margin must fit on both sides of threshold")
        if self.reason_count < 1:
            raise ValueError("reason_count must be positive")


@dataclass(frozen=True)
class Reason:
    feature: str
    direction: str
    contribution: float


@dataclass(frozen=True)
class DetectionResult:
    decision: str
    probability_synthetic: float
    decision_confidence: float
    raw_score: float
    threshold: float
    abstained: bool
    calibration: str
    reasons: tuple[Reason, ...]
    contributions: dict[str, float]


def decide(
    prediction: Prediction,
    *,
    calibrator: PlattCalibrator | None,
    policy: DecisionPolicy,
) -> DetectionResult:
    probability = calibrator.transform(prediction.raw_score) if calibrator else prediction.probability_synthetic
    abstained = abs(probability - policy.threshold) <= policy.abstain_margin
    if abstained:
        decision = "abstain"
    elif probability > policy.threshold:
        decision = "synthetic"
    else:
        decision = "camera_or_human"

    ordered = sorted(prediction.contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    reasons = tuple(
        Reason(feature=name, direction="synthetic" if contribution > 0.0 else "camera_or_human", contribution=contribution)
        for name, contribution in ordered[: policy.reason_count]
    )
    return DetectionResult(
        decision=decision,
        probability_synthetic=probability,
        decision_confidence=max(probability, 1.0 - probability),
        raw_score=prediction.raw_score,
        threshold=policy.threshold,
        abstained=abstained,
        calibration="platt" if calibrator else "uncalibrated",
        reasons=reasons,
        contributions=dict(prediction.contributions),
    )
