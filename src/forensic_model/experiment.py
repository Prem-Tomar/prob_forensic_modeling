"""Reproducible procedural experiment for end-to-end pipeline verification."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

from forensic_model.detector import ImageDetector
from forensic_model.features import FEATURE_VERSION
from forensic_model.image import RGBImage
from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.robustness import EvaluationSample, evaluate_postprocessing, evaluate_unseen_generators
from forensic_model.video import TEMPORAL_FEATURE_VERSION, TimedFrame, VideoClip, VideoDetector


SMOKE_SEED = 1729


def run_smoke_evaluation() -> dict[str, object]:
    """Exercise all library phases without making a real-world accuracy claim."""

    train_images = [_camera(index) for index in range(12)] + [_generated(index, "known-checker") for index in range(12)]
    image_detector = ImageDetector.train(train_images, [0] * 12 + [1] * 12)
    calibration_images = [_camera(100 + index) for index in range(6)] + [
        _generated(100 + index, "known-checker") for index in range(6)
    ]
    image_detector = image_detector.fit_calibrator(calibration_images, [0] * 6 + [1] * 6)

    in_distribution = [
        EvaluationSample(_camera(200 + index), 0, f"id-real-{index}") for index in range(8)
    ] + [
        EvaluationSample(_generated(200 + index, "known-checker"), 1, f"id-synthetic-{index}", "known-checker")
        for index in range(8)
    ]
    unseen = [EvaluationSample(_camera(300 + index), 0, f"holdout-real-{index}") for index in range(8)]
    for family in ("unseen-stripes", "unseen-blocks"):
        unseen.extend(
            EvaluationSample(_generated(300 + index, family), 1, f"{family}-{index}", family)
            for index in range(8)
        )

    in_distribution_metrics = _score_metrics(image_detector, in_distribution)
    unseen_metrics = evaluate_unseen_generators(
        image_detector,
        unseen,
        trained_generator_families={"known-checker"},
    )
    processing_metrics = evaluate_postprocessing(image_detector, unseen)
    unseen_labels = [sample.label for sample in unseen]
    unseen_probabilities = [image_detector.analyze_image(sample.image).probability_synthetic for sample in unseen]
    interval = grouped_bootstrap_interval(
        unseen_labels,
        unseen_probabilities,
        [sample.content_group for sample in unseen],
        statistic=auroc,
        resamples=300,
        seed=SMOKE_SEED,
    )

    video_report = _video_smoke(image_detector)
    return {
        "schema_version": 1,
        "report_kind": "procedural_smoke_evaluation",
        "claim_scope": "pipeline_mechanics_only",
        "seed": SMOKE_SEED,
        "implementation_sha256": _implementation_digest(),
        "feature_versions": {"image": FEATURE_VERSION, "video": TEMPORAL_FEATURE_VERSION},
        "image": {
            "in_distribution": in_distribution_metrics.to_dict(),
            "unseen_generators": {family: metrics.to_dict() for family, metrics in unseen_metrics.items()},
            "unseen_generator_auroc_interval": asdict(interval),
            "postprocessing": {name: metrics.to_dict() for name, metrics in processing_metrics.items()},
        },
        "provenance": {
            "evidence_states_covered_by_tests": ["absent", "valid", "invalid", "unsupported", "indeterminate"],
            "fusion": "late_fusion_with_conflict_abstention",
        },
        "video": video_report,
        "limitations": [
            "procedural fixtures validate software mechanics, not real-world detection quality",
            "no third-party dataset, pretrained weight, credential, or codec is used",
            "acceptance gates require approved local media and generator-family holdouts",
        ],
    }


def _score_metrics(detector: ImageDetector, samples: list[EvaluationSample]):
    probabilities = [detector.analyze_image(sample.image).probability_synthetic for sample in samples]
    return binary_metrics([sample.label for sample in samples], probabilities, threshold=detector.policy.threshold)


def _video_smoke(image_detector: ImageDetector) -> dict[str, object]:
    training = [_ordered_clip(f"train-real-{index}", index, 0, "known-checker") for index in range(6)] + [
        _ordered_clip(f"train-synthetic-{index}", index, 1, "known-checker") for index in range(6)
    ]
    video_detector = VideoDetector.train(image_detector, training, [0] * 6 + [1] * 6, max_frames=4)
    calibration = [_ordered_clip(f"cal-real-{index}", 50 + index, 0, "known-checker") for index in range(3)] + [
        _ordered_clip(f"cal-synthetic-{index}", 50 + index, 1, "known-checker") for index in range(3)
    ]
    video_detector = video_detector.fit_calibrator(calibration, [0] * 3 + [1] * 3)
    test = [_ordered_clip(f"test-real-{index}", 100 + index, 0, "unseen-stripes") for index in range(6)] + [
        _ordered_clip(f"test-synthetic-{index}", 100 + index, 1, "unseen-stripes") for index in range(6)
    ]
    labels = [0] * 6 + [1] * 6
    results = [video_detector.analyze_clip(clip) for clip in test]
    temporal = binary_metrics(labels, [result.decision.probability_synthetic for result in results])
    aggregation = binary_metrics(labels, [result.aggregation_probability for result in results])
    return {
        "holdout_family": "unseen-stripes",
        "temporal": temporal.to_dict(),
        "frame_aggregation_baseline": aggregation.to_dict(),
        "temporal_auroc_gain": temporal.auroc - aggregation.auroc,
        "clips": len(test),
    }


def _ordered_clip(clip_id: str, index: int, label: int, family: str) -> VideoClip:
    low = _camera(500 + index)
    high = _generated(500 + index, family)
    frames = (low, high, low, high) if label == 1 else (low, low, high, high)
    return VideoClip(clip_id, tuple(TimedFrame(float(position), image) for position, image in enumerate(frames)))


def _camera(index: int) -> RGBImage:
    base = 0.18 + 0.025 * (index % 9)
    rows = []
    for y in range(8):
        row = []
        for x in range(8):
            texture = 0.012 * math.sin((index + 1) * (x + 2) * (y + 1))
            red = _clip(base + 0.012 * x + texture)
            green = _clip(base + 0.009 * y - texture / 2.0)
            blue = _clip(base + 0.006 * (x + y) + texture / 3.0)
            row.append((red, green, blue))
        rows.append(row)
    return RGBImage.from_rows(rows)


def _generated(index: int, family: str) -> RGBImage:
    low = 0.12 + 0.015 * (index % 7)
    high = 0.82 - 0.012 * (index % 5)
    rows = []
    for y in range(8):
        row = []
        for x in range(8):
            if family == "known-checker":
                select_high = (x + y + index) % 2 == 0
            elif family == "unseen-stripes":
                select_high = (x + index) % 3 == 0
            elif family == "unseen-blocks":
                select_high = (x // 2 + y // 2 + index) % 2 == 0
            else:
                raise ValueError(f"unknown procedural family: {family}")
            value = high if select_high else low
            row.append((value, _clip(value * 0.94 + 0.02), _clip(value * 0.88 + 0.04)))
        rows.append(row)
    return RGBImage.from_rows(rows)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)
