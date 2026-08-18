"""High-level reusable image detector API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy, DetectionResult, decide
from forensic_model.features import FEATURE_VERSION, extract_features
from forensic_model.image import ImageDecoder, PPMDecoder, RGBImage
from forensic_model.model import LogisticModel, Prediction


@dataclass(frozen=True)
class ImageDetector:
    model: LogisticModel
    calibrator: PlattCalibrator | None = None
    policy: DecisionPolicy = DecisionPolicy()

    @classmethod
    def train(cls, images: Sequence[RGBImage], labels: Sequence[int]) -> "ImageDetector":
        return cls(LogisticModel.fit([extract_features(image) for image in images], labels))

    def predict_image(self, image: RGBImage) -> Prediction:
        return self.model.predict(extract_features(image))

    def analyze_image(self, image: RGBImage) -> DetectionResult:
        return decide(self.predict_image(image), calibrator=self.calibrator, policy=self.policy)

    def fit_calibrator(self, images: Sequence[RGBImage], labels: Sequence[int]) -> "ImageDetector":
        scores = [self.predict_image(image).raw_score for image in images]
        return ImageDetector(self.model, PlattCalibrator.fit(scores, labels), self.policy)

    def predict_file(self, path: Path, decoder: ImageDecoder | None = None) -> Prediction:
        return self.predict_image((decoder or PPMDecoder()).decode(path))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": 1,
            "model": self.model.to_dict(),
            "calibration": self.calibrator.to_dict() if self.calibrator else None,
            "policy": {
                "threshold": self.policy.threshold,
                "abstain_margin": self.policy.abstain_margin,
                "reason_count": self.policy.reason_count,
            },
        }

    @classmethod
    def load(cls, path: Path) -> "ImageDetector":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("detector artifact must be a JSON object")
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ImageDetector":
        if not isinstance(values, dict) or values.get("artifact_version") != 1:
            raise ValueError("unsupported detector artifact")
        model_values = values.get("model")
        calibration_values = values.get("calibration")
        policy_values = values.get("policy")
        if not isinstance(model_values, dict) or not isinstance(policy_values, dict):
            raise ValueError("model artifact must be a JSON object")
        calibrator = None
        if calibration_values is not None:
            if not isinstance(calibration_values, dict):
                raise ValueError("calibration artifact must be an object or null")
            calibrator = PlattCalibrator.from_dict(calibration_values)
        policy = DecisionPolicy(
            threshold=float(policy_values["threshold"]),
            abstain_margin=float(policy_values["abstain_margin"]),
            reason_count=int(policy_values["reason_count"]),
        )
        model = LogisticModel.from_dict(model_values)
        if model.feature_version != FEATURE_VERSION:
            raise ValueError("detector artifact does not use image features")
        return cls(model, calibrator, policy)
