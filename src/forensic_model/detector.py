"""High-level reusable image detector API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from forensic_model.features import FeatureVector, extract_features
from forensic_model.image import ImageDecoder, PPMDecoder, RGBImage
from forensic_model.model import LogisticModel, Prediction


@dataclass(frozen=True)
class ImageDetector:
    model: LogisticModel

    @classmethod
    def train(cls, images: Sequence[RGBImage], labels: Sequence[int]) -> "ImageDetector":
        return cls(LogisticModel.fit([extract_features(image) for image in images], labels))

    def predict_image(self, image: RGBImage) -> Prediction:
        return self.model.predict(extract_features(image))

    def predict_file(self, path: Path, decoder: ImageDecoder | None = None) -> Prediction:
        return self.predict_image((decoder or PPMDecoder()).decode(path))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ImageDetector":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("model artifact must be a JSON object")
        return cls(LogisticModel.from_dict(values))
