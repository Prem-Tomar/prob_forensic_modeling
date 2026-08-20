"""Audited public-video manifests for the optional neural video track."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from forensic_model.data_audit import sha256_file


@dataclass(frozen=True)
class DatasetAttribution:
    name: str
    source_url: str
    license_id: str
    citation: str


DAVIS_ATTRIBUTION = DatasetAttribution(
    name="DAVIS 2017",
    source_url="https://davischallenge.org/davis2017/code.html",
    license_id="CC-BY-NC-4.0",
    citation="Pont-Tuset, J. et al. (2017), The 2017 DAVIS Challenge on Video Object Segmentation.",
)

GENVIDBENCH_ATTRIBUTION = DatasetAttribution(
    name="GenVidBench",
    source_url="https://github.com/genvidbench/GenVidBench",
    license_id="CC-BY-NC-4.0",
    citation="Ni, Z.-L. et al. (2026), GenVidBench, AAAI.",
)

SORA_EXCLUSIONS = frozenset(
    {
        "Figure Status Update - OpenAI Speech-to-Speech Reasoning.mp4",
        "Sora | OpenAI -  - 2024-05-20 18-08-46.mp4",
        "Sora | OpenAI.mp4",
    }
)

KELING_EXCLUSIONS = frozenset(
    {
        "39462_1717765170_raw.mp4",
        "WeChat_20240608171016.mp4",
    }
)


@dataclass(frozen=True)
class VideoExample:
    path: Path
    label: int
    content_group: str
    source: str
    generator_family: str
    split: str
    media_type: str
    semantic_category: str = ""
    capture_device: str = ""


@dataclass(frozen=True)
class VideoAuditSummary:
    discovered: int
    retained: int
    duplicates_removed: int
    exclusions: int
    split_counts: Mapping[str, int]
    source_counts: Mapping[str, int]


@dataclass(frozen=True)
class VideoDatasetBundle:
    train: tuple[VideoExample, ...]
    validation: tuple[VideoExample, ...]
    test: tuple[VideoExample, ...]
    audit: VideoAuditSummary
    attributions: tuple[DatasetAttribution, ...]


def discover_video_benchmark(
    davis_root: Path,
    keling_root: Path,
    sora_root: Path,
    *,
    seed: str = "video-split-v1",
) -> VideoDatasetBundle:
    """Build source-grouped train/validation/test manifests without leakage."""

    rows = []
    exclusions = 0
    rows.extend(_discover_davis(davis_root, seed))
    keling_t2v, keling_t2v_exclusions = _discover_generated(
        keling_root / "keling" / "T2V",
        source="GenVidBench-Keling-T2V",
        family="keling",
        splits=("train", "validation"),
        seed=seed,
        excluded_names=KELING_EXCLUSIONS,
    )
    rows.extend(keling_t2v)
    exclusions += keling_t2v_exclusions
    keling_i2v, keling_i2v_exclusions = _discover_generated(
        keling_root / "keling" / "I2V",
        source="GenVidBench-Keling-I2V",
        family="keling",
        splits=("train", "validation"),
        seed=seed,
        excluded_names=frozenset(),
    )
    rows.extend(keling_i2v)
    exclusions += keling_i2v_exclusions
    sora, sora_exclusions = _discover_generated(
        sora_root,
        source="GenVidBench-Sora",
        family="sora",
        splits=("test",),
        seed=seed,
        excluded_names=SORA_EXCLUSIONS,
    )
    rows.extend(sora)
    exclusions += sora_exclusions
    if not rows:
        raise ValueError("video benchmark contains no examples")

    retained, duplicates = _deduplicate(rows)
    by_split: dict[str, list[VideoExample]] = {name: [] for name in ("train", "validation", "test")}
    for row in retained:
        by_split[row.split].append(row)
    for split, examples in by_split.items():
        if {example.label for example in examples} != {0, 1}:
            raise ValueError(f"{split} split must contain both labels")
    return VideoDatasetBundle(
        train=tuple(by_split["train"]),
        validation=tuple(by_split["validation"]),
        test=tuple(by_split["test"]),
        audit=VideoAuditSummary(
            discovered=len(rows) + exclusions,
            retained=len(retained),
            duplicates_removed=duplicates,
            exclusions=exclusions,
            split_counts=dict(Counter(row.split for row in retained)),
            source_counts=dict(Counter(row.source for row in retained)),
        ),
        attributions=(DAVIS_ATTRIBUTION, GENVIDBENCH_ATTRIBUTION),
    )


def _discover_davis(root: Path, seed: str) -> list[VideoExample]:
    frames_root = root / "JPEGImages" / "480p"
    train_names = _read_names(root / "ImageSets" / "2017" / "train.txt")
    validation_names = _read_names(root / "ImageSets" / "2017" / "val.txt")
    rows = []
    for name in train_names:
        rows.append(_frame_example(frames_root / name, "train"))
    for name in validation_names:
        digest = _hash_frame_directory(frames_root / name)
        split = "validation" if _fraction(seed, digest) < 0.5 else "test"
        rows.append(_frame_example(frames_root / name, split, digest=digest))
    return rows


def _discover_generated(
    root: Path,
    *,
    source: str,
    family: str,
    splits: Sequence[str],
    seed: str,
    excluded_names: frozenset[str],
) -> tuple[list[VideoExample], int]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".mp4", ".mov"} and not path.name.startswith("._")
    )
    rows = []
    exclusions = 0
    for path in paths:
        if path.name in excluded_names:
            exclusions += 1
            continue
        digest = sha256_file(path)
        split = splits[0]
        if len(splits) == 2:
            split = splits[0] if _fraction(seed, digest) < 0.8 else splits[1]
        rows.append(VideoExample(path, 1, digest, source, family, split, "video"))
    return rows, exclusions


def _frame_example(path: Path, split: str, *, digest: str | None = None) -> VideoExample:
    if not path.is_dir():
        raise ValueError(f"missing DAVIS frame sequence: {path}")
    identity = digest or _hash_frame_directory(path)
    return VideoExample(path, 0, identity, "DAVIS-2017", "", split, "frames")


def _deduplicate(rows: Iterable[VideoExample]) -> tuple[list[VideoExample], int]:
    retained: dict[str, VideoExample] = {}
    duplicates = 0
    for row in sorted(rows, key=lambda item: (item.content_group, item.path.as_posix())):
        previous = retained.get(row.content_group)
        if previous is None:
            retained[row.content_group] = row
            continue
        if previous.label != row.label:
            raise ValueError("identical video content has conflicting labels")
        if previous.split != row.split:
            raise ValueError("identical video content crosses dataset splits")
        duplicates += 1
    return sorted(retained.values(), key=lambda item: (item.split, item.source, item.path.as_posix())), duplicates


def _read_names(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise ValueError(f"missing DAVIS split manifest: {path}")
    names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not names:
        raise ValueError(f"empty DAVIS split manifest: {path}")
    return names


def _hash_frame_directory(path: Path) -> str:
    digest = hashlib.sha256()
    frames = sorted(path.glob("*.jpg"))
    if not frames:
        raise ValueError(f"frame sequence contains no JPEG images: {path}")
    for frame in frames:
        digest.update(frame.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(frame)))
    return digest.hexdigest()


def _fraction(seed: str, identity: str) -> float:
    value = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") / 2**64
