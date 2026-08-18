"""Temporal video analysis built on reusable decoded-frame sequences."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy, DetectionResult, decide
from forensic_model.detector import ImageDetector
from forensic_model.features import FeatureVector
from forensic_model.image import RGBImage
from forensic_model.model import LogisticModel, Prediction


TEMPORAL_FEATURE_VERSION = "temporal-score-v1"
TEMPORAL_FEATURE_NAMES = (
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "mean_absolute_delta",
    "max_absolute_delta",
    "score_trend",
    "alternating_delta_energy",
    "duration_seconds",
    "motion_residual_mean",
    "motion_residual_std",
    "luma_flicker_energy",
)


@dataclass(frozen=True)
class TimedFrame:
    timestamp_seconds: float
    image: RGBImage


@dataclass(frozen=True)
class VideoClip:
    clip_id: str
    frames: tuple[TimedFrame, ...]

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must not be empty")
        if len(self.frames) < 3:
            raise ValueError("video clips require at least three decoded frames")
        timestamps = [frame.timestamp_seconds for frame in self.frames]
        if any(not math.isfinite(timestamp) or timestamp < 0.0 for timestamp in timestamps):
            raise ValueError("timestamps must be finite and non-negative")
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")


class VideoDecoder(Protocol):
    def decode(self, path: Path) -> VideoClip:
        """Decode local video into ordered RGB frames without network access."""


@dataclass(frozen=True)
class FrameScore:
    timestamp_seconds: float
    probability_synthetic: float


@dataclass(frozen=True)
class VideoResult:
    clip_id: str
    decision: DetectionResult
    aggregation_probability: float
    frame_scores: tuple[FrameScore, ...]
    temporal_features: FeatureVector
    notable_timestamps: tuple[float, ...]


@dataclass(frozen=True)
class VideoDetector:
    image_detector: ImageDetector
    temporal_model: LogisticModel
    calibrator: PlattCalibrator | None = None
    policy: DecisionPolicy = DecisionPolicy()
    max_frames: int = 32

    @classmethod
    def train(
        cls,
        image_detector: ImageDetector,
        clips: Sequence[VideoClip],
        labels: Sequence[int],
        *,
        max_frames: int = 32,
    ) -> "VideoDetector":
        _validate_training(clips, labels, max_frames)
        vectors = [_clip_features(image_detector, clip, max_frames)[1] for clip in clips]
        return cls(image_detector, LogisticModel.fit(vectors, labels), max_frames=max_frames)

    def predict_clip(self, clip: VideoClip) -> Prediction:
        return self.temporal_model.predict(_clip_features(self.image_detector, clip, self.max_frames)[1])

    def fit_calibrator(self, clips: Sequence[VideoClip], labels: Sequence[int]) -> "VideoDetector":
        _validate_training(clips, labels, self.max_frames)
        scores = [self.predict_clip(clip).raw_score for clip in clips]
        return VideoDetector(
            self.image_detector,
            self.temporal_model,
            PlattCalibrator.fit(scores, labels),
            self.policy,
            self.max_frames,
        )

    def analyze_clip(self, clip: VideoClip) -> VideoResult:
        frame_scores, features = _clip_features(self.image_detector, clip, self.max_frames)
        prediction = self.temporal_model.predict(features)
        decision = decide(prediction, calibrator=self.calibrator, policy=self.policy)
        mean = sum(frame.probability_synthetic for frame in frame_scores) / len(frame_scores)
        notable = tuple(
            frame.timestamp_seconds
            for frame in sorted(frame_scores, key=lambda frame: abs(frame.probability_synthetic - mean), reverse=True)[:3]
        )
        return VideoResult(
            clip_id=clip.clip_id,
            decision=decision,
            aggregation_probability=mean,
            frame_scores=frame_scores,
            temporal_features=features,
            notable_timestamps=notable,
        )

    def save(self, path: Path) -> None:
        artifact = {
            "artifact_version": 1,
            "artifact_type": "video_detector",
            "image_detector": self.image_detector.to_dict(),
            "temporal_model": self.temporal_model.to_dict(),
            "calibration": self.calibrator.to_dict() if self.calibrator else None,
            "policy": {
                "threshold": self.policy.threshold,
                "abstain_margin": self.policy.abstain_margin,
                "reason_count": self.policy.reason_count,
            },
            "max_frames": self.max_frames,
        }
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VideoDetector":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or values.get("artifact_version") != 1 or values.get("artifact_type") != "video_detector":
            raise ValueError("unsupported video detector artifact")
        image_values = values.get("image_detector")
        temporal_values = values.get("temporal_model")
        calibration_values = values.get("calibration")
        policy_values = values.get("policy")
        if not isinstance(image_values, dict) or not isinstance(temporal_values, dict) or not isinstance(policy_values, dict):
            raise ValueError("video artifact components must be objects")
        temporal_model = LogisticModel.from_dict(temporal_values)
        if temporal_model.feature_version != TEMPORAL_FEATURE_VERSION:
            raise ValueError("video artifact does not use temporal features")
        calibrator = None
        if calibration_values is not None:
            if not isinstance(calibration_values, dict):
                raise ValueError("video calibration must be an object or null")
            calibrator = PlattCalibrator.from_dict(calibration_values)
        return cls(
            image_detector=ImageDetector.from_dict(image_values),
            temporal_model=temporal_model,
            calibrator=calibrator,
            policy=DecisionPolicy(
                threshold=float(policy_values["threshold"]),
                abstain_margin=float(policy_values["abstain_margin"]),
                reason_count=int(policy_values["reason_count"]),
            ),
            max_frames=int(values["max_frames"]),
        )


def sample_frames(clip: VideoClip, max_frames: int) -> tuple[TimedFrame, ...]:
    """Select endpoints, strongest scene changes, then uniform coverage."""

    if max_frames < 3:
        raise ValueError("max_frames must be at least three")
    if len(clip.frames) <= max_frames:
        return clip.frames

    selected = {0, len(clip.frames) - 1}
    scene_slots = max(1, (max_frames - 2) // 2)
    changes = sorted(
        ((_scene_change(clip.frames[index - 1].image, clip.frames[index].image), index) for index in range(1, len(clip.frames))),
        key=lambda item: (-item[0], item[1]),
    )
    selected.update(index for _, index in changes[:scene_slots])

    target_count = max_frames
    uniform_candidates = [round(step * (len(clip.frames) - 1) / (target_count - 1)) for step in range(target_count)]
    for index in uniform_candidates:
        if len(selected) >= target_count:
            break
        selected.add(index)
    for index in range(len(clip.frames)):
        if len(selected) >= target_count:
            break
        selected.add(index)
    chosen = sorted(selected)
    return tuple(clip.frames[index] for index in chosen)


def extract_temporal_features(
    frame_scores: Sequence[FrameScore],
    frames: Sequence[TimedFrame],
) -> FeatureVector:
    if len(frame_scores) < 3:
        raise ValueError("temporal features require at least three frame scores")
    if len(frames) != len(frame_scores):
        raise ValueError("decoded frames must align with frame scores")
    scores = [frame.probability_synthetic for frame in frame_scores]
    if any(not math.isfinite(score) or score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("frame probabilities must be finite and within [0, 1]")
    mean = sum(scores) / len(scores)
    standard_deviation = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
    deltas = [current - previous for previous, current in zip(scores, scores[1:])]
    duration = frame_scores[-1].timestamp_seconds - frame_scores[0].timestamp_seconds
    trend = (scores[-1] - scores[0]) / duration if duration > 0.0 else 0.0
    alternating = abs(sum((1.0 if index % 2 == 0 else -1.0) * delta for index, delta in enumerate(deltas))) / len(deltas)
    motion_residuals = [_motion_compensated_residual(previous.image, current.image) for previous, current in zip(frames, frames[1:])]
    motion_mean = sum(motion_residuals) / len(motion_residuals)
    motion_std = math.sqrt(sum((residual - motion_mean) ** 2 for residual in motion_residuals) / len(motion_residuals))
    luma_means = [sum(_luma(pixel) for pixel in frame.image.pixels) / len(frame.image.pixels) for frame in frames]
    luma_deltas = [current - previous for previous, current in zip(luma_means, luma_means[1:])]
    flicker = abs(sum((1.0 if index % 2 == 0 else -1.0) * delta for index, delta in enumerate(luma_deltas))) / len(luma_deltas)
    return FeatureVector(
        names=TEMPORAL_FEATURE_NAMES,
        values=(
            mean,
            standard_deviation,
            min(scores),
            max(scores),
            sum(abs(delta) for delta in deltas) / len(deltas),
            max(abs(delta) for delta in deltas),
            trend,
            alternating,
            duration,
            motion_mean,
            motion_std,
            flicker,
        ),
        version=TEMPORAL_FEATURE_VERSION,
    )


def _clip_features(
    image_detector: ImageDetector,
    clip: VideoClip,
    max_frames: int,
) -> tuple[tuple[FrameScore, ...], FeatureVector]:
    sampled = sample_frames(clip, max_frames)
    scores = tuple(
        FrameScore(frame.timestamp_seconds, image_detector.analyze_image(frame.image).probability_synthetic)
        for frame in sampled
    )
    return scores, extract_temporal_features(scores, sampled)


def _scene_change(first: RGBImage, second: RGBImage) -> float:
    if first.width == second.width and first.height == second.height:
        return sum(
            abs(_luma(first_pixel) - _luma(second_pixel))
            for first_pixel, second_pixel in zip(first.pixels, second.pixels)
        ) / len(first.pixels)
    return abs(sum(_luma(pixel) for pixel in first.pixels) / len(first.pixels) - sum(_luma(pixel) for pixel in second.pixels) / len(second.pixels))


def _luma(pixel: tuple[float, float, float]) -> float:
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def _motion_compensated_residual(first: RGBImage, second: RGBImage) -> float:
    if first.width != second.width or first.height != second.height:
        return _scene_change(first, second)
    candidates = []
    for shift_y in (-1, 0, 1):
        for shift_x in (-1, 0, 1):
            differences = []
            for y in range(max(0, -shift_y), min(first.height, first.height - shift_y)):
                for x in range(max(0, -shift_x), min(first.width, first.width - shift_x)):
                    differences.append(abs(_luma(first.pixel(x, y)) - _luma(second.pixel(x + shift_x, y + shift_y))))
            if differences:
                candidates.append(sum(differences) / len(differences))
    return min(candidates)


def _validate_training(clips: Sequence[VideoClip], labels: Sequence[int], max_frames: int) -> None:
    if len(clips) != len(labels) or len(clips) < 2:
        raise ValueError("clips and labels must have equal length of at least two")
    if set(labels) != {0, 1}:
        raise ValueError("video labels must contain both 0 and 1")
    if max_frames < 3:
        raise ValueError("max_frames must be at least three")
