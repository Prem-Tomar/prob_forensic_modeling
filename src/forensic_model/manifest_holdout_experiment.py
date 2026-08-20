"""Evaluate a frozen image detector on governed, content-matched holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader

from forensic_model.checkpoint_digest import semantic_checkpoint_sha256
from forensic_model.manifest_image import ManifestImageAudit, load_manifest_image_corpus
from forensic_model.metrics import auroc, binary_metrics, classification_diagnostics, grouped_bootstrap_interval
from forensic_model.neural_data import PillowImageDataset, evaluation_transform
from forensic_model.neural_inference import CalibratedNeuralImageDetector
from forensic_model.slice_evaluation import evaluate_slices


def run_manifest_holdout_experiment(
    manifest: Path,
    media_root: Path,
    checkpoint: Path,
    *,
    output: Path,
    image_size: int = 32,
    batch_size: int = 64,
    abstain_margin: float = 0.05,
    bootstrap_resamples: int = 500,
    seed: int = 20260818,
) -> dict[str, object]:
    """Score every matched generator family without fitting or retuning."""

    if (
        image_size < 8
        or batch_size <= 0
        or not 0.0 <= abstain_margin < 0.5
        or bootstrap_resamples < 20
        or seed < 0
    ):
        raise ValueError("invalid holdout evaluation configuration")
    corpus = load_manifest_image_corpus(manifest, media_root=media_root)
    matched = corpus.matched_generator_holdouts()
    if not matched:
        raise ValueError("manifest contains no matched generator holdouts")
    detector = CalibratedNeuralImageDetector.load(checkpoint, abstain_margin=abstain_margin)
    family_reports: dict[str, dict[str, Any]] = {}
    for family, holdout in matched.items():
        dataset = PillowImageDataset(holdout.examples, evaluation_transform(image_size))
        probabilities, labels, groups = _score(detector, dataset, batch_size=batch_size)
        metrics = binary_metrics(labels, probabilities, threshold=detector.threshold)
        interval = grouped_bootstrap_interval(
            labels,
            probabilities,
            groups,
            statistic=auroc,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        control_sources = [row.source.casefold() for row in holdout.real_controls] * 2
        semantic_categories = [row.semantic_category.casefold() for row in holdout.examples]
        capture_devices = [row.capture_device.casefold() for row in holdout.real_controls] * 2
        slice_metadata_complete = bool(
            all(semantic_categories)
            and all(capture_devices)
            and all(control_sources)
        )
        family_reports[family] = {
            "content_groups_per_class": len(holdout.real_controls),
            "metrics": metrics.to_dict(),
            "diagnostics": classification_diagnostics(
                labels,
                probabilities,
                threshold=detector.threshold,
                abstain_margin=detector.abstain_margin,
            ).to_dict(),
            "auroc_grouped_bootstrap_95_percent": asdict(interval),
            "slice_coverage": {
                "metadata_complete": slice_metadata_complete,
                "control_sources": sorted(set(control_sources)),
                "semantic_categories": sorted(set(semantic_categories) - {""}),
                "capture_devices": sorted(set(capture_devices) - {""}),
            },
            "slices": {
                "control_source": evaluate_slices(
                    labels,
                    probabilities,
                    groups,
                    control_sources,
                    threshold=detector.threshold,
                    abstain_margin=detector.abstain_margin,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                ),
                "semantic_category": evaluate_slices(
                    labels,
                    probabilities,
                    groups,
                    semantic_categories,
                    threshold=detector.threshold,
                    abstain_margin=detector.abstain_margin,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                ),
                "capture_device": evaluate_slices(
                    labels,
                    probabilities,
                    groups,
                    capture_devices,
                    threshold=detector.threshold,
                    abstain_margin=detector.abstain_margin,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                ),
            },
        }
    eligible = len(family_reports) >= 3 and all(
        int(report["content_groups_per_class"]) >= 1000
        and bool(report["slice_coverage"]["metadata_complete"])
        for report in family_reports.values()
    )
    gate_results = {
        "auroc_at_least_0_85": all(float(report["metrics"]["auroc"]) >= 0.85 for report in family_reports.values()),
        "balanced_accuracy_at_least_0_75": all(
            float(report["metrics"]["balanced_accuracy"]) >= 0.75 for report in family_reports.values()
        ),
        "brier_at_most_0_18": all(float(report["metrics"]["brier"]) <= 0.18 for report in family_reports.values()),
        "ece_at_most_0_05": all(
            float(report["metrics"]["expected_calibration_error"]) <= 0.05
            for report in family_reports.values()
        ),
    }
    aurocs = [float(report["metrics"]["auroc"]) for report in family_reports.values()]
    report: dict[str, object] = {
        "claim_scope": "frozen_matched_generator_holdout_not_production_assurance",
        "configuration": {
            "image_size": image_size,
            "batch_size": batch_size,
            "abstain_margin": abstain_margin,
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
        },
        "model": {
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_semantic_sha256": semantic_checkpoint_sha256(checkpoint),
            "training_or_tuning_during_evaluation": False,
            "threshold": detector.threshold,
        },
        "data_audit": _audit_dict(corpus.audit),
        "generator_families": family_reports,
        "aggregate": {
            "family_count": len(family_reports),
            "macro_auroc": sum(aurocs) / len(aurocs),
            "worst_family_auroc": min(aurocs),
        },
        "acceptance": {
            "eligible": eligible,
            "minimum_generator_families": 3,
            "minimum_content_groups_per_class_per_family": 1000,
            "complete_slice_metadata_required": True,
            "gate_results": gate_results,
            "all_gates_passed": eligible and all(gate_results.values()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-evaluate-manifest-holdouts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--abstain-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_manifest_holdout_experiment(
        options.manifest,
        options.media_root,
        options.checkpoint,
        output=options.output,
        image_size=options.image_size,
        batch_size=options.batch_size,
        abstain_margin=options.abstain_margin,
        bootstrap_resamples=options.bootstrap_resamples,
        seed=options.seed,
    )
    return 0


def _score(
    detector: CalibratedNeuralImageDetector,
    dataset: PillowImageDataset,
    *,
    batch_size: int,
) -> tuple[list[float], list[int], list[str]]:
    probabilities: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    for images, targets, identities in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        probabilities.extend(result.probability_synthetic for result in detector.predict_tensors(images))
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    return probabilities, labels, groups


def _audit_dict(audit: ManifestImageAudit) -> dict[str, Any]:
    return {
        "manifest_sha256": audit.manifest_sha256,
        "verified_files": audit.verified_files,
        "split_counts": dict(audit.split_counts),
        "label_counts": dict(audit.label_counts),
        "generator_counts": dict(audit.generator_counts),
        "source_license_counts": dict(audit.source_license_counts),
        "attributions": [asdict(attribution) for attribution in audit.attributions],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
