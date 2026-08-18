"""Dependency-free binary ranking, decision, and calibration metrics."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class BinaryMetrics:
    count: int
    positives: int
    negatives: int
    auroc: float
    auprc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    brier: float
    log_loss: float
    expected_calibration_error: float
    false_positive_rate_at_90_recall: float
    recall_at_1_percent_false_positive_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    successful_resamples: int


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    calibration_bins: int = 15,
) -> BinaryMetrics:
    _validate(labels, probabilities)
    if not 0.0 < threshold < 1.0 or calibration_bins < 2:
        raise ValueError("invalid metric configuration")

    positives = sum(labels)
    negatives = len(labels) - positives
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(prediction == 1 and label == 1 for prediction, label in zip(predictions, labels))
    true_negative = sum(prediction == 0 and label == 0 for prediction, label in zip(predictions, labels))
    false_positive = negatives - true_negative
    sensitivity = true_positive / positives
    specificity = true_negative / negatives
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = sensitivity
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    epsilon = 1e-15
    clipped = [min(max(probability, epsilon), 1.0 - epsilon) for probability in probabilities]
    log_loss = -sum(label * math.log(probability) + (1 - label) * math.log(1.0 - probability) for label, probability in zip(labels, clipped)) / len(labels)

    return BinaryMetrics(
        count=len(labels),
        positives=positives,
        negatives=negatives,
        auroc=_auroc(labels, probabilities),
        auprc=_average_precision(labels, probabilities),
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        recall=recall,
        f1=f1,
        brier=sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels),
        log_loss=log_loss,
        expected_calibration_error=_equal_mass_ece(labels, probabilities, calibration_bins),
        false_positive_rate_at_90_recall=_false_positive_rate_at_recall(labels, probabilities, 0.9),
        recall_at_1_percent_false_positive_rate=_recall_at_false_positive_rate(labels, probabilities, 0.01),
    )


def grouped_bootstrap_interval(
    labels: Sequence[int],
    probabilities: Sequence[float],
    groups: Sequence[str],
    *,
    statistic: Callable[[Sequence[int], Sequence[float]], float],
    resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    _validate(labels, probabilities)
    if len(groups) != len(labels) or len(set(groups)) < 2:
        raise ValueError("groups must align with rows and contain at least two identities")
    if resamples < 20 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")

    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_rows.setdefault(group, []).append(index)
    identities = sorted(group_rows)
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        selected = [randomizer.choice(identities) for _ in identities]
        indices = [index for group in selected for index in group_rows[group]]
        sampled_labels = [labels[index] for index in indices]
        if set(sampled_labels) != {0, 1}:
            continue
        estimates.append(statistic(sampled_labels, [probabilities[index] for index in indices]))
    if len(estimates) < max(10, resamples // 2):
        raise ValueError("too few valid bootstrap resamples")
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=statistic(labels, probabilities),
        lower=_percentile(estimates, tail),
        upper=_percentile(estimates, 1.0 - tail),
        confidence=confidence,
        successful_resamples=len(estimates),
    )


def auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(labels, probabilities)
    return _auroc(labels, probabilities)


def _validate(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if len(labels) != len(probabilities) or len(labels) < 2:
        raise ValueError("labels and probabilities must have equal length of at least two")
    if set(labels) != {0, 1}:
        raise ValueError("metrics require both binary classes")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("probabilities must be finite and within [0, 1]")


def _auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positive += sum(label for _, label in ordered[index:end])
        false_positive += (end - index) - sum(label for _, label in ordered[index:end])
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _equal_mass_ece(labels: Sequence[int], probabilities: Sequence[float], bins: int) -> float:
    ordered = sorted(zip(probabilities, labels))
    bin_count = min(bins, len(ordered))
    error = 0.0
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        rows = ordered[start:end]
        mean_probability = sum(probability for probability, _ in rows) / len(rows)
        mean_label = sum(label for _, label in rows) / len(rows)
        error += len(rows) / len(ordered) * abs(mean_probability - mean_label)
    return error


def _operating_points(labels: Sequence[int], probabilities: Sequence[float]) -> list[tuple[float, float]]:
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    negatives = len(labels) - positives
    points = [(0.0, 0.0)]
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positive += sum(label for _, label in ordered[index:end])
        false_positive += (end - index) - sum(label for _, label in ordered[index:end])
        points.append((false_positive / negatives, true_positive / positives))
        index = end
    return points


def _false_positive_rate_at_recall(labels: Sequence[int], probabilities: Sequence[float], target: float) -> float:
    return min(false_positive_rate for false_positive_rate, recall in _operating_points(labels, probabilities) if recall >= target)


def _recall_at_false_positive_rate(labels: Sequence[int], probabilities: Sequence[float], maximum: float) -> float:
    return max(recall for false_positive_rate, recall in _operating_points(labels, probabilities) if false_positive_rate <= maximum)


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
