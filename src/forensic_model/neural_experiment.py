"""Executable, reproducible CIFAKE-to-SynthScars image experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.neural import save_neural_checkpoint
from forensic_model.neural_data import (
    SYNTHSCARS_ATTRIBUTION,
    PillowImageDataset,
    discover_cifake,
    discover_synthscars_test,
    evaluation_transform,
    image_split_digest,
    training_transform,
)
from forensic_model.neural_training import (
    TrainedNeuralDetector,
    TrainingConfig,
    recalibrate_neural_detector,
    score_neural_detector,
    train_neural_detector,
)
from forensic_model.neural_robustness import select_frequency_weight


def run_real_image_experiment(
    cifake_root: Path,
    synthscars_root: Path,
    *,
    output: Path,
    checkpoint: Path,
    training_config: TrainingConfig = TrainingConfig(),
) -> dict[str, object]:
    """Train from scratch, stress the test split, and evaluate an unseen source."""

    bundle = discover_cifake(cifake_root)
    train = PillowImageDataset(bundle.train, training_transform())
    validation = PillowImageDataset(bundle.validation, evaluation_transform())
    detector = train_neural_detector(train, validation, training_config=training_config)
    validation_sets = {
        operation: PillowImageDataset(bundle.validation, evaluation_transform(operation=operation))
        for operation in ("clean", "jpeg30", "blur1", "resize50")
    }
    branch_selection = select_frequency_weight(
        detector.model,
        validation_sets,
        batch_size=training_config.batch_size,
    )
    detector = recalibrate_neural_detector(
        detector,
        validation_sets["clean"],
        frequency_weight=branch_selection.frequency_weight,
    )

    stress_metrics = {}
    clean_details: tuple[list[float], list[int], list[str]] | None = None
    for operation in ("clean", "jpeg30", "blur1", "resize50"):
        dataset = PillowImageDataset(bundle.test, evaluation_transform(operation=operation))
        details = _calibrated_scores(detector, dataset)
        probabilities, labels, _ = details
        stress_metrics[operation] = binary_metrics(labels, probabilities, threshold=detector.threshold).to_dict()
        if operation == "clean":
            clean_details = details
    if clean_details is None:
        raise RuntimeError("clean evaluation was not produced")
    clean_probabilities, clean_labels, clean_groups = clean_details
    interval = grouped_bootstrap_interval(
        clean_labels,
        clean_probabilities,
        clean_groups,
        statistic=auroc,
        resamples=500,
        seed=training_config.seed,
    )

    synthscars = discover_synthscars_test(synthscars_root)
    real_rows = tuple(example for example in bundle.test if example.label == 0)[: len(synthscars)]
    unseen_dataset = PillowImageDataset(real_rows + synthscars, evaluation_transform())
    unseen_probabilities, unseen_labels, _ = _calibrated_scores(detector, unseen_dataset)
    unseen_metrics = binary_metrics(unseen_labels, unseen_probabilities, threshold=detector.threshold)

    explanations = _branch_summary(
        detector.model,
        PillowImageDataset(bundle.test[:256], evaluation_transform()),
        frequency_weight=detector.frequency_weight,
    )
    metadata = detector.metadata()
    metadata.update({"dataset": "CIFAKE", "split_seed": "cifake-split-v1"})
    save_neural_checkpoint(checkpoint, detector.model, metadata=metadata)

    report: dict[str, object] = {
        "claim_scope": "research_baseline_not_production_assurance",
        "training": "from_scratch",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "device": "cpu",
        },
        "configuration": asdict(training_config),
        "model": {
            "architecture": "spatial-frequency-v1",
            "parameters": sum(parameter.numel() for parameter in detector.model.parameters()),
            "pretrained_weights": False,
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "data_audit": asdict(bundle.audit),
        "split_membership_sha256": image_split_digest(bundle),
        "attribution": [asdict(item) for item in bundle.attributions + (SYNTHSCARS_ATTRIBUTION,)],
        "training_history": [
            {
                "epoch": epoch.epoch,
                "train_loss": epoch.train_loss,
                "validation_auroc": epoch.validation_auroc,
            }
            for epoch in detector.history
        ],
        "calibration": detector.calibrator.to_dict(),
        "robust_branch_selection": asdict(branch_selection),
        "threshold_selected_on_validation": detector.threshold,
        "cifake_test": stress_metrics,
        "clean_auroc_grouped_bootstrap_95_percent": asdict(interval),
        "unseen_synthscars_with_cifake_real_control": unseen_metrics.to_dict(),
        "unseen_evaluation_limit": (
            "SynthScars contains synthetic-only holdout images; pairing with CIFAKE real images introduces content, "
            "resolution, and source confounds, so this metric is diagnostic rather than a deployment claim."
        ),
        "explanation_summary": explanations,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-train-image")
    parser.add_argument("--cifake-root", type=Path, required=True)
    parser.add_argument("--synthscars-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_real_image_experiment(
        options.cifake_root,
        options.synthscars_root,
        output=options.output,
        checkpoint=options.checkpoint,
        training_config=TrainingConfig(seed=options.seed, epochs=options.epochs, batch_size=options.batch_size),
    )
    return 0


def _calibrated_scores(
    detector: TrainedNeuralDetector,
    dataset: PillowImageDataset,
) -> tuple[list[float], list[int], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=detector.training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    scores, labels, groups = score_neural_detector(
        detector.model, loader, frequency_weight=detector.frequency_weight
    )
    return [detector.calibrator.transform(score) for score in scores], labels, groups


@torch.inference_mode()
def _branch_summary(
    model: torch.nn.Module,
    dataset: PillowImageDataset,
    *,
    frequency_weight: float,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    spatial = []
    frequency = []
    model.eval()
    for images, _, _ in loader:
        evidence = model.forward_evidence(images)
        spatial.extend(evidence.spatial_contribution.tolist())
        frequency.extend(evidence.frequency_contribution.tolist())
    deployed_frequency = [frequency_weight * value for value in frequency]
    return {
        "sample_count": len(spatial),
        "selected_frequency_weight": frequency_weight,
        "mean_spatial_contribution": sum(spatial) / len(spatial),
        "mean_frequency_contribution": sum(frequency) / len(frequency),
        "mean_deployed_frequency_contribution": sum(deployed_frequency) / len(deployed_frequency),
        "mean_absolute_spatial_contribution": sum(abs(value) for value in spatial) / len(spatial),
        "mean_absolute_frequency_contribution": sum(abs(value) for value in frequency) / len(frequency),
        "mean_absolute_deployed_frequency_contribution": (
            sum(abs(value) for value in deployed_frequency) / len(deployed_frequency)
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
