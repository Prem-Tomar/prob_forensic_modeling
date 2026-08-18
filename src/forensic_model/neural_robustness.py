"""Validation-only selection of robust spatial-frequency evidence blends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from forensic_model.metrics import auroc
from forensic_model.neural import SpatialFrequencyDetector


@dataclass(frozen=True)
class BranchBlendSelection:
    """A frequency-branch weight selected without consulting test examples."""

    frequency_weight: float
    worst_case_auroc: float
    mean_auroc: float
    operation_aurocs: Mapping[str, float]


def select_frequency_weight(
    model: SpatialFrequencyDetector,
    validation_sets: Mapping[str, Dataset],
    *,
    candidate_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    batch_size: int = 256,
) -> BranchBlendSelection:
    """Maximize worst-slice AUROC across named validation transformations."""

    if not validation_sets:
        raise ValueError("at least one validation set is required")
    weights = tuple(float(weight) for weight in candidate_weights)
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("candidate weights must be within [0, 1]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    evidence = {
        operation: _score_branches(model, dataset, batch_size)
        for operation, dataset in sorted(validation_sets.items())
    }
    selections = []
    for weight in weights:
        operation_aurocs = {}
        for operation, (spatial, frequency, labels) in evidence.items():
            probabilities = torch.sigmoid(spatial + weight * frequency).tolist()
            operation_aurocs[operation] = auroc(labels, probabilities)
        values = tuple(operation_aurocs.values())
        selections.append(
            BranchBlendSelection(weight, min(values), sum(values) / len(values), operation_aurocs)
        )
    return max(selections, key=lambda item: (item.worst_case_auroc, item.mean_auroc, -item.frequency_weight))


@torch.inference_mode()
def _score_branches(
    model: SpatialFrequencyDetector,
    dataset: Dataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    model.eval()
    spatial = []
    frequency = []
    labels = []
    for images, targets, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch = model.forward_evidence(images)
        spatial.append(batch.spatial_contribution.cpu())
        frequency.append(batch.frequency_contribution.cpu())
        labels.extend(int(value) for value in targets.tolist())
    if not labels:
        raise ValueError("validation sets must not be empty")
    return torch.cat(spatial), torch.cat(frequency), labels
