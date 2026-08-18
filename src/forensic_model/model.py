"""Deterministic learned baseline with serializable parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from forensic_model.features import FEATURE_VERSION, FeatureVector


@dataclass(frozen=True)
class Prediction:
    probability_synthetic: float
    raw_score: float
    contributions: dict[str, float]


@dataclass(frozen=True)
class LogisticModel:
    """Standardized logistic regression trained with full-batch gradient descent."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    feature_version: str = FEATURE_VERSION

    @classmethod
    def fit(
        cls,
        features: Sequence[FeatureVector],
        labels: Sequence[int],
        *,
        learning_rate: float = 0.15,
        epochs: int = 800,
        l2: float = 0.01,
    ) -> "LogisticModel":
        if len(features) != len(labels) or len(features) < 2:
            raise ValueError("features and labels must have the same length of at least two")
        if set(labels) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")
        if learning_rate <= 0.0 or epochs <= 0 or l2 < 0.0:
            raise ValueError("learning_rate and epochs must be positive; l2 must be non-negative")
        names = features[0].names
        if any(vector.names != names or vector.version != FEATURE_VERSION for vector in features):
            raise ValueError("all feature vectors must use the same supported schema")

        columns = tuple(tuple(vector.values[index] for vector in features) for index in range(len(names)))
        means = tuple(sum(column) / len(column) for column in columns)
        scales = tuple(max(_standard_deviation(column, mean), 1e-9) for column, mean in zip(columns, means))
        matrix = tuple(tuple((value - mean) / scale for value, mean, scale in zip(vector.values, means, scales)) for vector in features)

        weights = [0.0] * len(names)
        bias = 0.0
        count = len(matrix)
        for _ in range(epochs):
            probabilities = [_sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias) for row in matrix]
            errors = [probability - label for probability, label in zip(probabilities, labels)]
            gradients = [sum(error * row[index] for error, row in zip(errors, matrix)) / count + l2 * weights[index] for index in range(len(weights))]
            bias_gradient = sum(errors) / count
            weights = [weight - learning_rate * gradient for weight, gradient in zip(weights, gradients)]
            bias -= learning_rate * bias_gradient

        return cls(names, means, scales, tuple(weights), bias)

    def predict(self, features: FeatureVector) -> Prediction:
        if features.names != self.feature_names or features.version != self.feature_version:
            raise ValueError("feature schema does not match the model")
        standardized = tuple((value - mean) / scale for value, mean, scale in zip(features.values, self.means, self.scales))
        contributions = {name: weight * value for name, weight, value in zip(self.feature_names, self.weights, standardized)}
        score = self.bias + sum(contributions.values())
        return Prediction(_sigmoid(score), score, contributions)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "standardized_logistic_regression",
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "LogisticModel":
        if values.get("model_type") != "standardized_logistic_regression":
            raise ValueError("unsupported model type")
        if values.get("feature_version") != FEATURE_VERSION:
            raise ValueError("unsupported feature version")
        model = cls(
            feature_names=tuple(str(value) for value in _list(values, "feature_names")),
            means=tuple(float(value) for value in _list(values, "means")),
            scales=tuple(float(value) for value in _list(values, "scales")),
            weights=tuple(float(value) for value in _list(values, "weights")),
            bias=float(values["bias"]),
        )
        lengths = {len(model.feature_names), len(model.means), len(model.scales), len(model.weights)}
        numeric = model.means + model.scales + model.weights + (model.bias,)
        if lengths != {len(model.feature_names)} or not model.feature_names:
            raise ValueError("model parameter lengths must match and be non-empty")
        if any(not math.isfinite(value) for value in numeric) or any(scale <= 0.0 for scale in model.scales):
            raise ValueError("model parameters must be finite and scales must be positive")
        return model


def _list(values: dict[str, object], key: str) -> list[object]:
    value = values.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _standard_deviation(values: Iterable[float], mean: float) -> float:
    values = tuple(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
