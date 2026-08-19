"""Executable cross-generator experiment for the neural temporal detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import av
import PIL
import torch
from torch.utils.data import DataLoader

from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.neural_video import save_temporal_checkpoint
from forensic_model.neural_video_data import VideoDatasetBundle, discover_video_benchmark
from forensic_model.neural_video_training import (
    TemporalTrainingConfig,
    TrainedTemporalDetector,
    score_temporal_detector,
    train_temporal_detector,
)
from forensic_model.video_decode import VideoTensorDataset


def run_real_video_experiment(
    davis_root: Path,
    keling_root: Path,
    sora_root: Path,
    *,
    output: Path,
    checkpoint: Path,
    training_config: TemporalTrainingConfig = TemporalTrainingConfig(),
    frame_count: int = 8,
    image_size: int = 32,
) -> dict[str, object]:
    """Train on DAVIS/Keling T2V+I2V and test once on held-out DAVIS/Sora sources."""

    bundle = discover_video_benchmark(davis_root, keling_root, sora_root)
    train = VideoTensorDataset(
        bundle.train,
        frame_count=frame_count,
        image_size=image_size,
        cache=True,
    )
    validation = VideoTensorDataset(
        bundle.validation,
        frame_count=frame_count,
        image_size=image_size,
        cache=True,
    )
    detector = train_temporal_detector(train, validation, training_config=training_config)

    stress_metrics = {}
    clean_details = None
    operations = {
        "clean": ("clean", frame_count),
        "jpeg30": ("jpeg30", frame_count),
        "blur1": ("blur1", frame_count),
        "resize50": ("resize50", frame_count),
        "sparse4": ("clean", 4),
    }
    for name, (operation, frames) in operations.items():
        dataset = VideoTensorDataset(
            bundle.test,
            frame_count=frames,
            image_size=image_size,
            operation=operation,
            cache=True,
        )
        details = _calibrated_scores(detector, dataset)
        probabilities, labels, _ = details
        stress_metrics[name] = binary_metrics(labels, probabilities, threshold=detector.threshold).to_dict()
        if name == "clean":
            clean_details = details
    if clean_details is None:
        raise RuntimeError("clean temporal evaluation was not produced")
    probabilities, labels, groups = clean_details
    interval = grouped_bootstrap_interval(
        labels,
        probabilities,
        groups,
        statistic=auroc,
        resamples=500,
        seed=training_config.seed,
    )
    source_summary = _source_summary(bundle, probabilities)
    explanations = _evidence_summary(
        detector,
        VideoTensorDataset(bundle.test, frame_count=frame_count, image_size=image_size, cache=True),
    )
    generated_test = tuple(example for example in bundle.test if example.media_type == "video")
    generated_duration = sum(_duration_seconds(example.path) for example in generated_test)
    generated_started = time.monotonic()
    _calibrated_scores(
        detector,
        VideoTensorDataset(generated_test, frame_count=frame_count, image_size=image_size),
    )
    generated_elapsed = time.monotonic() - generated_started

    metadata = detector.metadata()
    metadata.update(
        {
            "dataset": "DAVIS-2017+GenVidBench-Keling-T2V+I2V",
            "training_data_license": "CC-BY-NC-4.0",
            "usage_scope": "non-commercial-research",
            "frame_count": frame_count,
            "image_size": image_size,
        }
    )
    save_temporal_checkpoint(checkpoint, detector.model, metadata=metadata)
    report: dict[str, object] = {
        "claim_scope": "research_baseline_not_production_assurance",
        "training": "from_scratch",
        "artifact_scope": "non-commercial-research_due_to_training_data_license",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pillow": PIL.__version__,
            "pyav": av.__version__,
            "device": "cpu",
        },
        "configuration": {
            **asdict(training_config),
            "frame_count": frame_count,
            "image_size": image_size,
        },
        "model": {
            "architecture": "temporal-residual-v1",
            "parameters": sum(parameter.numel() for parameter in detector.model.parameters()),
            "pretrained_weights": False,
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "data_audit": asdict(bundle.audit),
        "attribution": [asdict(item) for item in bundle.attributions],
        "training_history": [asdict(epoch) for epoch in detector.history],
        "calibration": detector.calibrator.to_dict(),
        "threshold_selected_on_validation": detector.threshold,
        "cross_generator_test": stress_metrics,
        "clean_auroc_grouped_bootstrap_95_percent": asdict(interval),
        "source_summary": source_summary,
        "explanation_summary": explanations,
        "timing": {
            "generated_test_duration_seconds": generated_duration,
            "generated_decode_inference_seconds": generated_elapsed,
            "seconds_per_input_minute": generated_elapsed / (generated_duration / 60.0),
        },
        "evaluation_limit": (
            "Generated and real labels come from different source datasets. Sora is unseen during training, but "
            "content, codec, duration, and collection-source confounds remain; results are diagnostic, not a "
            "deployment or generator-attribution claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-train-video")
    parser.add_argument("--davis-root", type=Path, required=True)
    parser.add_argument("--keling-root", type=Path, required=True)
    parser.add_argument("--sora-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_real_video_experiment(
        options.davis_root,
        options.keling_root,
        options.sora_root,
        output=options.output,
        checkpoint=options.checkpoint,
        training_config=TemporalTrainingConfig(
            seed=options.seed,
            epochs=options.epochs,
            batch_size=options.batch_size,
            patience=options.patience,
        ),
        frame_count=options.frame_count,
        image_size=options.image_size,
    )
    return 0


def _calibrated_scores(
    detector: TrainedTemporalDetector,
    dataset: VideoTensorDataset,
) -> tuple[list[float], list[int], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=detector.training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    scores, labels, groups = score_temporal_detector(detector.model, loader)
    return [detector.calibrator.transform(score) for score in scores], labels, groups


def _source_summary(bundle: VideoDatasetBundle, probabilities: Sequence[float]) -> dict[str, dict[str, float | int]]:
    if len(bundle.test) != len(probabilities):
        raise ValueError("test examples and probabilities must align")
    by_source: dict[str, list[float]] = {}
    for example, probability in zip(bundle.test, probabilities):
        by_source.setdefault(example.source, []).append(probability)
    return {
        source: {
            "count": len(values),
            "mean_probability_synthetic": sum(values) / len(values),
        }
        for source, values in sorted(by_source.items())
    }


@torch.inference_mode()
def _evidence_summary(
    detector: TrainedTemporalDetector,
    dataset: VideoTensorDataset,
) -> dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=detector.training_config.batch_size, shuffle=False, num_workers=0)
    frame = []
    temporal = []
    detector.model.eval()
    for clips, _, _ in loader:
        evidence = detector.model.forward_evidence(clips)
        frame.extend(evidence.frame_contribution.tolist())
        temporal.extend(evidence.temporal_contribution.tolist())
    return {
        "sample_count": len(frame),
        "mean_absolute_frame_contribution": sum(abs(value) for value in frame) / len(frame),
        "mean_absolute_temporal_contribution": sum(abs(value) for value in temporal) / len(temporal),
    }


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
