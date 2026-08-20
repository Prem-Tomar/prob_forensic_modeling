"""Reusable frozen-threshold metrics for governed evaluation slices."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from forensic_model.metrics import auroc, binary_metrics, classification_diagnostics, grouped_bootstrap_interval


def evaluate_slices(
    labels: Sequence[int],
    probabilities: Sequence[float],
    groups: Sequence[str],
    values: Sequence[str],
    *,
    threshold: float,
    abstain_margin: float = 0.0,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate every non-empty slice without fitting or threshold changes."""

    if (
        not 0.0 < threshold < 1.0
        or not 0.0 <= abstain_margin < min(threshold, 1.0 - threshold)
        or bootstrap_resamples < 20
        or seed < 0
    ):
        raise ValueError("invalid slice evaluation configuration")
    if not (len(labels) == len(probabilities) == len(groups) == len(values)):
        raise ValueError("slice values must align with scored examples")
    reports: dict[str, dict[str, Any]] = {}
    for value in sorted(set(values) - {""}):
        indices = [index for index, candidate in enumerate(values) if candidate == value]
        slice_labels = [labels[index] for index in indices]
        slice_probabilities = [probabilities[index] for index in indices]
        slice_groups = [groups[index] for index in indices]
        common = {
            "sample_count": len(indices),
            "content_groups": len(set(slice_groups)),
        }
        if set(slice_labels) != {0, 1}:
            reports[value] = {**common, "status": "insufficient_label_coverage"}
            continue
        metrics = binary_metrics(slice_labels, slice_probabilities, threshold=threshold)
        diagnostics = classification_diagnostics(
            slice_labels,
            slice_probabilities,
            threshold=threshold,
            abstain_margin=abstain_margin,
        ).to_dict()
        if len(set(slice_groups)) < 2:
            reports[value] = {
                **common,
                "status": "insufficient_group_coverage",
                "metrics": metrics.to_dict(),
                "diagnostics": diagnostics,
                "auroc_grouped_bootstrap_95_percent": None,
            }
            continue
        interval = grouped_bootstrap_interval(
            slice_labels,
            slice_probabilities,
            slice_groups,
            statistic=auroc,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        reports[value] = {
            **common,
            "status": "evaluated",
            "metrics": metrics.to_dict(),
            "diagnostics": diagnostics,
            "auroc_grouped_bootstrap_95_percent": asdict(interval),
        }
    return reports
