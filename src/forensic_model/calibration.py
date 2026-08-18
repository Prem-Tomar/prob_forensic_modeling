"""Probability calibration fitted only on held-out model scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PlattCalibrator:
    """One-dimensional logistic calibration for raw classifier scores."""

    slope: float
    intercept: float

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int],
        *,
        learning_rate: float = 0.05,
        epochs: int = 1200,
        l2: float = 0.001,
    ) -> "PlattCalibrator":
        if len(scores) != len(labels) or len(scores) < 4:
            raise ValueError("calibration requires equally sized scores and labels with at least four rows")
        if set(labels) != {0, 1}:
            raise ValueError("calibration labels must contain both 0 and 1")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("calibration scores must be finite")
        if learning_rate <= 0.0 or epochs <= 0 or l2 < 0.0:
            raise ValueError("invalid calibration optimization settings")

        slope = 1.0
        intercept = 0.0
        count = len(scores)
        for _ in range(epochs):
            probabilities = [_sigmoid(slope * score + intercept) for score in scores]
            errors = [probability - label for probability, label in zip(probabilities, labels)]
            slope_gradient = sum(error * score for error, score in zip(errors, scores)) / count + l2 * slope
            intercept_gradient = sum(errors) / count
            slope -= learning_rate * slope_gradient
            intercept -= learning_rate * intercept_gradient
        return cls(slope=slope, intercept=intercept)

    def transform(self, raw_score: float) -> float:
        if not math.isfinite(raw_score):
            raise ValueError("raw score must be finite")
        return _sigmoid(self.slope * raw_score + self.intercept)

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "platt", "slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "PlattCalibrator":
        if values.get("method") != "platt":
            raise ValueError("unsupported calibration method")
        calibrator = cls(slope=float(values["slope"]), intercept=float(values["intercept"]))
        if not math.isfinite(calibrator.slope) or not math.isfinite(calibrator.intercept):
            raise ValueError("calibration parameters must be finite")
        return calibrator


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
