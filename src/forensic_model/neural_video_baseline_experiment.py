"""Reproducible matched frame-aggregation evaluation for video holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import av

from forensic_model.checkpoint_digest import semantic_checkpoint_sha256
from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.neural_video_baseline import (
    FrozenFrameAggregator,
    fit_clip_aggregation,
    score_validated_frame_aggregation,
)
from forensic_model.neural_video_data import discover_video_benchmark
from forensic_model.video_decode import VideoTensorDataset


def run_frame_aggregation_experiment(
    davis_root: Path,
    keling_root: Path,
    sora_root: Path,
    image_checkpoint: Path,
    *,
    output: Path,
    frame_count: int = 8,
    image_size: int = 32,
    batch_size: int = 8,
    seed: int = 20260818,
) -> dict[str, object]:
    """Freeze the image model, tune clip calibration on validation, then test."""

    started = time.monotonic()
    bundle = discover_video_benchmark(davis_root, keling_root, sora_root)
    audit_elapsed = time.monotonic() - started
    frozen = FrozenFrameAggregator.load(image_checkpoint)
    validation = VideoTensorDataset(
        bundle.validation,
        frame_count=frame_count,
        image_size=image_size,
        cache=True,
    )
    fit_started = time.monotonic()
    calibrated = fit_clip_aggregation(frozen, validation, batch_size=batch_size)
    validation_elapsed = time.monotonic() - fit_started
    test = VideoTensorDataset(
        bundle.test,
        frame_count=frame_count,
        image_size=image_size,
        cache=True,
    )
    test_started = time.monotonic()
    probabilities, labels, groups = score_validated_frame_aggregation(
        calibrated,
        test,
        batch_size=batch_size,
    )
    test_elapsed = time.monotonic() - test_started
    metrics = binary_metrics(labels, probabilities, threshold=calibrated.threshold)
    interval = grouped_bootstrap_interval(
        labels,
        probabilities,
        groups,
        statistic=auroc,
        resamples=500,
        seed=seed,
    )
    generated_test = tuple(example for example in bundle.test if example.media_type == "video")
    generated_duration = sum(_duration_seconds(example.path) for example in generated_test)
    generated_started = time.monotonic()
    score_validated_frame_aggregation(
        calibrated,
        VideoTensorDataset(generated_test, frame_count=frame_count, image_size=image_size),
        batch_size=batch_size,
    )
    generated_elapsed = time.monotonic() - generated_started
    report: dict[str, object] = {
        "claim_scope": "matched_research_baseline_not_production_assurance",
        "model_training": "frozen_image_detector_trained_from_scratch",
        "video_training": False,
        "configuration": {
            "frame_count": frame_count,
            "image_size": image_size,
            "batch_size": batch_size,
            "seed": seed,
        },
        "image_checkpoint_sha256": _sha256(image_checkpoint),
        "image_checkpoint_semantic_sha256": semantic_checkpoint_sha256(image_checkpoint),
        "data_audit": asdict(bundle.audit),
        "attribution": [asdict(item) for item in bundle.attributions],
        "clip_calibration": calibrated.clip_calibrator.to_dict(),
        "threshold_selected_on_validation": calibrated.threshold,
        "cross_generator_test": metrics.to_dict(),
        "auroc_grouped_bootstrap_95_percent": asdict(interval),
        "timing": {
            "audit_seconds": audit_elapsed,
            "validation_decode_fit_seconds": validation_elapsed,
            "test_decode_inference_seconds": test_elapsed,
            "generated_test_duration_seconds": generated_duration,
            "generated_decode_inference_seconds": generated_elapsed,
            "seconds_per_input_minute": generated_elapsed / (generated_duration / 60.0),
            "end_to_end_seconds": time.monotonic() - started,
        },
        "evaluation_limit": (
            "The frozen image detector was trained on CIFAKE, while video labels come from DAVIS and GenVidBench. "
            "Clip calibration uses only the video validation split; source and content confounds remain."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-evaluate-video-frames")
    parser.add_argument("--davis-root", type=Path, required=True)
    parser.add_argument("--keling-root", type=Path, required=True)
    parser.add_argument("--sora-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_frame_aggregation_experiment(
        options.davis_root,
        options.keling_root,
        options.sora_root,
        options.image_checkpoint,
        output=options.output,
        frame_count=options.frame_count,
        image_size=options.image_size,
        batch_size=options.batch_size,
        seed=options.seed,
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is None:
            raise ValueError(f"video duration is unavailable: {path}")
        return float(container.duration / av.time_base)


if __name__ == "__main__":
    raise SystemExit(main())
