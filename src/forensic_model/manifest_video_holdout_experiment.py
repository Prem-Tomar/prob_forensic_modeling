"""Compare frozen temporal and frame evidence on matched video holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader

from forensic_model.checkpoint_digest import semantic_checkpoint_sha256
from forensic_model.manifest_video import ManifestVideoAudit, load_manifest_video_corpus
from forensic_model.metrics import auroc, binary_metrics, classification_diagnostics, grouped_bootstrap_interval
from forensic_model.neural_video_baseline import (
    ValidatedFrameAggregator,
    score_validated_frame_aggregation,
)
from forensic_model.slice_evaluation import evaluate_slices
from forensic_model.temporal_inference import CalibratedTemporalDetector
from forensic_model.video_decode import VideoTensorDataset


def run_manifest_video_holdout_experiment(
    manifest: Path,
    media_root: Path,
    temporal_checkpoint: Path,
    image_checkpoint: Path,
    frame_validation_report: Path,
    *,
    output: Path,
    frame_count: int = 8,
    image_size: int = 32,
    batch_size: int = 8,
    abstain_margin: float = 0.05,
    bootstrap_resamples: int = 500,
    seed: int = 20260818,
) -> dict[str, object]:
    """Evaluate frozen models per family without holdout fitting or threshold changes."""

    if (
        frame_count < 3
        or image_size < 8
        or batch_size <= 0
        or not 0.0 <= abstain_margin < 0.5
        or bootstrap_resamples < 20
        or seed < 0
    ):
        raise ValueError("invalid matched video evaluation configuration")
    corpus = load_manifest_video_corpus(manifest, media_root=media_root)
    matched = corpus.matched_generator_holdouts()
    if not matched:
        raise ValueError("manifest contains no matched video generator holdouts")
    temporal = CalibratedTemporalDetector.load(temporal_checkpoint, abstain_margin=abstain_margin)
    frame = ValidatedFrameAggregator.load(image_checkpoint, frame_validation_report)
    family_reports: dict[str, dict[str, Any]] = {}
    for family, holdout in matched.items():
        dataset = VideoTensorDataset(
            holdout.examples,
            frame_count=frame_count,
            image_size=image_size,
            cache=True,
        )
        temporal_probabilities, labels, groups, explanation = _score_temporal(
            temporal,
            dataset,
            batch_size=batch_size,
        )
        frame_probabilities, frame_labels, frame_groups = score_validated_frame_aggregation(
            frame,
            dataset,
            batch_size=batch_size,
        )
        if (frame_labels, frame_groups) != (labels, groups):
            raise RuntimeError("temporal and frame evaluations are not row-aligned")
        temporal_metrics = binary_metrics(labels, temporal_probabilities, threshold=temporal.threshold)
        frame_metrics = binary_metrics(labels, frame_probabilities, threshold=frame.threshold)
        control_sources = [row.source.casefold() for row in holdout.real_controls] * 2
        semantic_categories = [row.semantic_category.casefold() for row in holdout.examples]
        capture_devices = [row.capture_device.casefold() for row in holdout.real_controls] * 2
        slice_dimensions = {
            "control_source": control_sources,
            "semantic_category": semantic_categories,
            "capture_device": capture_devices,
        }
        family_reports[family] = {
            "content_groups_per_class": len(holdout.real_controls),
            "slice_coverage": {
                "metadata_complete": bool(
                    all(control_sources)
                    and all(semantic_categories)
                    and all(capture_devices)
                ),
                "control_sources": sorted(set(control_sources)),
                "semantic_categories": sorted(set(semantic_categories) - {""}),
                "capture_devices": sorted(set(capture_devices) - {""}),
            },
            "temporal": {
                "metrics": temporal_metrics.to_dict(),
                "diagnostics": classification_diagnostics(
                    labels,
                    temporal_probabilities,
                    threshold=temporal.threshold,
                    abstain_margin=temporal.abstain_margin,
                ).to_dict(),
                "auroc_grouped_bootstrap_95_percent": asdict(
                    grouped_bootstrap_interval(
                        labels,
                        temporal_probabilities,
                        groups,
                        statistic=auroc,
                        resamples=bootstrap_resamples,
                        seed=seed,
                    )
                ),
                "explanation_summary": explanation,
            },
            "frame_aggregation": {
                "metrics": frame_metrics.to_dict(),
                "diagnostics": classification_diagnostics(
                    labels,
                    frame_probabilities,
                    threshold=frame.threshold,
                    abstain_margin=abstain_margin,
                ).to_dict(),
                "auroc_grouped_bootstrap_95_percent": asdict(
                    grouped_bootstrap_interval(
                        labels,
                        frame_probabilities,
                        groups,
                        statistic=auroc,
                        resamples=bootstrap_resamples,
                        seed=seed + 1,
                    )
                ),
            },
            "slices": {
                "temporal": {
                    dimension: evaluate_slices(
                        labels,
                        temporal_probabilities,
                        groups,
                        values,
                        threshold=temporal.threshold,
                        abstain_margin=temporal.abstain_margin,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=seed,
                    )
                    for dimension, values in slice_dimensions.items()
                },
                "frame_aggregation": {
                    dimension: evaluate_slices(
                        labels,
                        frame_probabilities,
                        groups,
                        values,
                        threshold=frame.threshold,
                        abstain_margin=abstain_margin,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=seed + 1,
                    )
                    for dimension, values in slice_dimensions.items()
                },
            },
            "temporal_minus_frame_auroc": temporal_metrics.auroc - frame_metrics.auroc,
        }
    temporal_aurocs = [float(value["temporal"]["metrics"]["auroc"]) for value in family_reports.values()]
    frame_aurocs = [float(value["frame_aggregation"]["metrics"]["auroc"]) for value in family_reports.values()]
    temporal_macro = sum(temporal_aurocs) / len(temporal_aurocs)
    frame_macro = sum(frame_aurocs) / len(frame_aurocs)
    eligible = len(family_reports) >= 3 and all(
        int(value["content_groups_per_class"]) >= 1000
        and bool(value["slice_coverage"]["metadata_complete"])
        for value in family_reports.values()
    )
    report: dict[str, object] = {
        "claim_scope": "frozen_matched_video_holdout_not_production_assurance",
        "configuration": {
            "frame_count": frame_count,
            "image_size": image_size,
            "batch_size": batch_size,
            "abstain_margin": abstain_margin,
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
        },
        "models": {
            "temporal": {
                "checkpoint_sha256": _sha256(temporal_checkpoint),
                "checkpoint_semantic_sha256": semantic_checkpoint_sha256(temporal_checkpoint),
                "threshold": temporal.threshold,
            },
            "frame_aggregation": {
                "image_checkpoint_sha256": _sha256(image_checkpoint),
                "image_checkpoint_semantic_sha256": semantic_checkpoint_sha256(image_checkpoint),
                "validation_report_sha256": _sha256(frame_validation_report),
                "threshold": frame.threshold,
            },
            "training_calibration_or_threshold_tuning_during_evaluation": False,
        },
        "data_audit": _audit_dict(corpus.audit),
        "generator_families": family_reports,
        "aggregate": {
            "family_count": len(family_reports),
            "temporal_macro_auroc": temporal_macro,
            "frame_aggregation_macro_auroc": frame_macro,
            "temporal_minus_frame_macro_auroc": temporal_macro - frame_macro,
            "temporal_worst_family_auroc": min(temporal_aurocs),
            "frame_aggregation_worst_family_auroc": min(frame_aurocs),
        },
        "phase5_adoption": {
            "eligible": eligible,
            "minimum_generator_families": 3,
            "minimum_content_groups_per_class_per_family": 1000,
            "complete_slice_metadata_required": True,
            "temporal_macro_auroc_exceeds_frame": temporal_macro > frame_macro,
            "temporal_auroc_exceeds_frame_for_every_family": all(
                temporal_score > frame_score
                for temporal_score, frame_score in zip(temporal_aurocs, frame_aurocs)
            ),
            "passed": eligible
            and temporal_macro > frame_macro
            and all(
                temporal_score > frame_score
                for temporal_score, frame_score in zip(temporal_aurocs, frame_aurocs)
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-evaluate-manifest-video-holdouts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--frame-validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--abstain-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_manifest_video_holdout_experiment(
        options.manifest,
        options.media_root,
        options.temporal_checkpoint,
        options.image_checkpoint,
        options.frame_validation_report,
        output=options.output,
        frame_count=options.frame_count,
        image_size=options.image_size,
        batch_size=options.batch_size,
        abstain_margin=options.abstain_margin,
        bootstrap_resamples=options.bootstrap_resamples,
        seed=options.seed,
    )
    return 0


def _score_temporal(
    detector: CalibratedTemporalDetector,
    dataset: VideoTensorDataset,
    *,
    batch_size: int,
) -> tuple[list[float], list[int], list[str], dict[str, float | int]]:
    probabilities: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    frame_contributions: list[float] = []
    temporal_contributions: list[float] = []
    for clips, targets, identities in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        results = detector.predict_tensors(clips)
        probabilities.extend(result.probability_synthetic for result in results)
        frame_contributions.extend(result.frame_contribution for result in results)
        temporal_contributions.extend(result.temporal_contribution for result in results)
        labels.extend(int(value) for value in targets.tolist())
        groups.extend(identities)
    return probabilities, labels, groups, {
        "sample_count": len(probabilities),
        "mean_absolute_frame_contribution": sum(abs(value) for value in frame_contributions) / len(probabilities),
        "mean_absolute_temporal_contribution": sum(abs(value) for value in temporal_contributions) / len(probabilities),
    }


def _audit_dict(audit: ManifestVideoAudit) -> dict[str, Any]:
    return {
        "manifest_sha256": audit.manifest_sha256,
        "verified_files": audit.verified_files,
        "split_counts": dict(audit.split_counts),
        "label_counts": dict(audit.label_counts),
        "generator_counts": dict(audit.generator_counts),
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
