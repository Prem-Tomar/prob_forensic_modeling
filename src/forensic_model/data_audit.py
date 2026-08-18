"""Deterministic duplicate auditing before a dataset manifest is accepted."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class DataAuditError(ValueError):
    """Raised when raw dataset identities cannot be partitioned safely."""


@dataclass(frozen=True)
class CandidateSample:
    sample_id: str
    path: Path
    label: str
    sha256: str
    perceptual_hash: str = ""


@dataclass(frozen=True)
class PartitionedSample:
    candidate: CandidateSample
    content_group: str
    split: str


@dataclass(frozen=True)
class AuditSummary:
    input_samples: int
    retained_samples: int
    exact_duplicate_groups: int
    exact_duplicates_removed: int
    perceptual_collision_groups: int
    split_counts: Mapping[str, int]


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash exact file bytes without loading a media file into memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_candidates(
    candidates: Iterable[CandidateSample],
    *,
    seed: str,
    train_percent: int = 70,
    validation_percent: int = 15,
) -> tuple[list[PartitionedSample], AuditSummary]:
    """Remove exact duplicates and assign stable, content-group-safe splits.

    Split membership is derived from the seed and exact content hash, so input
    ordering and filesystem traversal cannot change an experiment partition.
    Perceptual collisions are reported for review rather than silently merged:
    a compact hash collision is evidence for inspection, not proof of identity.
    """

    rows = list(candidates)
    if not rows:
        raise DataAuditError("dataset contains no candidates")
    if not seed:
        raise DataAuditError("split seed must not be empty")
    if not 1 <= train_percent < 100 or not 1 <= validation_percent < 100:
        raise DataAuditError("split percentages must be within [1, 99]")
    if train_percent + validation_percent >= 100:
        raise DataAuditError("train and validation percentages must leave a test split")

    by_hash: dict[str, list[CandidateSample]] = {}
    seen_ids: set[str] = set()
    perceptual_groups: dict[str, set[str]] = {}
    for candidate in rows:
        _validate_candidate(candidate)
        if candidate.sample_id in seen_ids:
            raise DataAuditError(f"duplicate sample_id: {candidate.sample_id}")
        seen_ids.add(candidate.sample_id)
        exact_hash = candidate.sha256.lower()
        by_hash.setdefault(exact_hash, []).append(candidate)
        if candidate.perceptual_hash:
            perceptual_groups.setdefault(candidate.perceptual_hash.lower(), set()).add(exact_hash)

    retained: list[PartitionedSample] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for exact_hash, group in sorted(by_hash.items()):
        labels = {candidate.label for candidate in group}
        if len(labels) != 1:
            ids = ", ".join(sorted(candidate.sample_id for candidate in group))
            raise DataAuditError(f"exact duplicates have conflicting labels: {ids}")
        canonical = min(group, key=lambda candidate: (str(candidate.path), candidate.sample_id))
        split = _stable_split(seed, exact_hash, train_percent, validation_percent)
        retained.append(PartitionedSample(candidate=canonical, content_group=exact_hash, split=split))
        split_counts[split] += 1

    exact_groups = [group for group in by_hash.values() if len(group) > 1]
    perceptual_collisions = [hashes for hashes in perceptual_groups.values() if len(hashes) > 1]
    return retained, AuditSummary(
        input_samples=len(rows),
        retained_samples=len(retained),
        exact_duplicate_groups=len(exact_groups),
        exact_duplicates_removed=sum(len(group) - 1 for group in exact_groups),
        perceptual_collision_groups=len(perceptual_collisions),
        split_counts=split_counts,
    )


def _stable_split(seed: str, content_group: str, train_percent: int, validation_percent: int) -> str:
    digest = hashlib.sha256(f"{seed}:{content_group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + validation_percent:
        return "validation"
    return "test"


def _validate_candidate(candidate: CandidateSample) -> None:
    if not candidate.sample_id:
        raise DataAuditError("sample_id must not be empty")
    if candidate.label not in {"camera_or_human", "synthetic"}:
        raise DataAuditError(f"{candidate.sample_id}: unsupported label {candidate.label!r}")
    if len(candidate.sha256) != 64 or any(character not in "0123456789abcdef" for character in candidate.sha256.lower()):
        raise DataAuditError(f"{candidate.sample_id}: invalid sha256")
    if candidate.perceptual_hash and any(
        character not in "0123456789abcdef" for character in candidate.perceptual_hash.lower()
    ):
        raise DataAuditError(f"{candidate.sample_id}: invalid perceptual_hash")
