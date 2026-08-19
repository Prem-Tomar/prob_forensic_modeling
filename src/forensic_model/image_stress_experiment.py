"""Paired post-processing evaluation for frozen image detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from forensic_model.detector import ImageDetector
from forensic_model.feature_baseline import extract_summary_feature_matrix, score_feature_model
from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.neural_data import (
    IMAGE_STRESS_OPERATIONS,
    PillowImageDataset,
    discover_cifake,
    evaluation_transform,
    image_split_digest,
)
from forensic_model.neural_inference import CalibratedNeuralImageDetector


OPERATION_DESCRIPTIONS = {
    "clean": "decode only",
    "jpeg30": "JPEG recompression at quality 30",
    "webp30": "WebP recompression at quality 30 with method 6",
    "resize50": "bilinear 50 percent downscale followed by upscale",
    "crop80": "center crop retaining 80 percent per axis followed by bicubic resize",
    "blur1": "Gaussian blur with radius 1.0",
    "sharpen2": "Pillow sharpness enhancement factor 2.0",
    "noise02": "deterministic additive RGB noise with maximum amplitude 0.02",
    "gamma08": "per-channel gamma exponent 0.8",
    "color70": "color saturation factor 0.7",
    "screenshot": "88 percent inset on a light canvas followed by JPEG quality 85",
    "metadata_strip": "lossless PNG pixel round trip without source metadata",
}


def run_image_stress_experiment(
    cifake_root: Path,
    *,
    neural_checkpoint: Path,
    feature_artifact: Path,
    output: Path,
    batch_size: int = 512,
    bootstrap_resamples: int = 200,
    cpu_threads: int = 8,
    seed: int = 20260818,
) -> dict[str, object]:
    """Evaluate two frozen calibrated detectors on identical transformed rows."""

    if batch_size <= 0 or cpu_threads <= 0:
        raise ValueError("batch size and CPU threads must be positive")
    if bootstrap_resamples < 20 or seed < 0:
        raise ValueError("bootstrap resamples must be at least 20 and seed must be non-negative")
    bundle = discover_cifake(cifake_root)
    neural = CalibratedNeuralImageDetector.load(neural_checkpoint, abstain_margin=0.0)
    feature = ImageDetector.load(feature_artifact)
    if feature.calibrator is None:
        raise ValueError("feature artifact must include calibration")

    old_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    probabilities: dict[str, dict[str, list[float]]] = {"neural": {}, "feature": {}}
    slices: dict[str, dict[str, Any]] = {}
    try:
        for operation_index, operation in enumerate(IMAGE_STRESS_OPERATIONS):
            neural_scores, feature_scores, labels, groups = _score_models(
                neural,
                feature,
                PillowImageDataset(bundle.test, evaluation_transform(operation=operation)),
                batch_size=batch_size,
            )
            probabilities["neural"][operation] = neural_scores
            probabilities["feature"][operation] = feature_scores
            slices[operation] = {
                "description": OPERATION_DESCRIPTIONS[operation],
                "neural": _measured_slice(
                    labels,
                    neural_scores,
                    groups,
                    threshold=neural.threshold,
                    resamples=bootstrap_resamples,
                    seed=seed + operation_index * 2,
                ),
                "feature": _measured_slice(
                    labels,
                    feature_scores,
                    groups,
                    threshold=feature.policy.threshold,
                    resamples=bootstrap_resamples,
                    seed=seed + operation_index * 2 + 1,
                ),
            }
    finally:
        torch.set_num_threads(old_threads)

    for operation in IMAGE_STRESS_OPERATIONS:
        for model_name in ("neural", "feature"):
            slices[operation][model_name]["paired_probability_shift_from_clean"] = _paired_shift(
                probabilities[model_name]["clean"],
                probabilities[model_name][operation],
            )
        slices[operation]["neural_minus_feature_auroc"] = (
            slices[operation]["neural"]["metrics"]["auroc"]
            - slices[operation]["feature"]["metrics"]["auroc"]
        )

    report: dict[str, object] = {
        "claim_scope": "frozen_matched_postprocessing_stress_test_not_production_assurance",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "device": "cpu",
        },
        "configuration": {
            "batch_size": batch_size,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_confidence": 0.95,
            "cpu_threads": cpu_threads,
            "seed": seed,
        },
        "models": {
            "neural": {
                "architecture": "spatial-frequency-v1",
                "checkpoint_sha256": _sha256(neural_checkpoint),
                "threshold": neural.threshold,
                "frequency_weight": neural.frequency_weight,
            },
            "feature": {
                "architecture": "standardized_logistic_regression",
                "artifact_sha256": _sha256(feature_artifact),
                "threshold": feature.policy.threshold,
            },
        },
        "data_audit": asdict(bundle.audit),
        "split_membership_sha256": image_split_digest(bundle),
        "sample_count_per_slice": len(bundle.test),
        "operation_order": list(IMAGE_STRESS_OPERATIONS),
        "slices": slices,
        "limitations": [
            "All rows come from the single CIFAKE generator family and are not evidence of cross-generator generalization.",
            "Transforms are deterministic reference implementations, not an exhaustive editor or platform matrix.",
            "The same frozen thresholds and calibrators are used throughout; no stress-slice tuning is performed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-evaluate-image-stress")
    parser.add_argument("--cifake-root", type=Path, required=True)
    parser.add_argument("--neural-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_image_stress_experiment(
        options.cifake_root,
        neural_checkpoint=options.neural_checkpoint,
        feature_artifact=options.feature_artifact,
        output=options.output,
        batch_size=options.batch_size,
        bootstrap_resamples=options.bootstrap_resamples,
        cpu_threads=options.cpu_threads,
        seed=options.seed,
    )
    return 0


@torch.inference_mode()
def _score_models(
    neural: CalibratedNeuralImageDetector,
    feature: ImageDetector,
    dataset: PillowImageDataset,
    *,
    batch_size: int,
) -> tuple[list[float], list[float], list[int], list[str]]:
    neural_probabilities: list[float] = []
    feature_probabilities: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    if feature.calibrator is None:
        raise ValueError("feature artifact must include calibration")
    calibrator = feature.calibrator
    for images, targets, identities in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        neural_probabilities.extend(result.probability_synthetic for result in neural.predict_tensors(images))
        feature_scores = score_feature_model(feature.model, extract_summary_feature_matrix(images))
        feature_probabilities.extend(calibrator.transform(score) for score in feature_scores)
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    if not labels:
        raise ValueError("stress dataset must not be empty")
    return neural_probabilities, feature_probabilities, labels, groups


def _measured_slice(
    labels: list[int],
    probabilities: list[float],
    groups: list[str],
    *,
    threshold: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    return {
        "metrics": binary_metrics(labels, probabilities, threshold=threshold).to_dict(),
        "auroc_grouped_bootstrap_95_percent": asdict(
            grouped_bootstrap_interval(
                labels,
                probabilities,
                groups,
                statistic=auroc,
                resamples=resamples,
                seed=seed,
            )
        ),
    }


def _paired_shift(clean: Sequence[float], transformed: Sequence[float]) -> dict[str, float]:
    if len(clean) != len(transformed) or not clean:
        raise ValueError("paired score vectors must be aligned and non-empty")
    changes = [after - before for before, after in zip(clean, transformed)]
    return {
        "mean_signed": sum(changes) / len(changes),
        "mean_absolute": sum(abs(value) for value in changes) / len(changes),
        "maximum_absolute": max(abs(value) for value in changes),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
