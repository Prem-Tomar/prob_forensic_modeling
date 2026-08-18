"""Dataset manifest parsing and leakage checks.

The validator is intentionally independent of model code: a run must prove that
its evidence is licensed and partitioned correctly before training begins.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ALLOWED_LABELS = frozenset({"camera_or_human", "synthetic"})
ALLOWED_SPLITS = frozenset(
    {
        "train",
        "validation",
        "calibration",
        "test",
        "generator_holdout",
        "source_holdout",
        "postprocess",
        "video_holdout",
    }
)
REQUIRED_COLUMNS = (
    "sample_id",
    "content_group",
    "path",
    "sha256",
    "label",
    "split",
    "source",
    "source_type",
    "license_id",
    "generator_family",
    "transformation",
)
UNKNOWN_LICENSES = frozenset({"", "unknown", "unverified", "none", "n/a"})


class ManifestError(ValueError):
    """Raised when a manifest cannot support a valid experiment."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    content_group: str
    path: str
    sha256: str
    label: str
    split: str
    source: str
    source_type: str
    license_id: str
    generator_family: str
    transformation: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Sample":
        return cls(**{column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS})


def load_manifest(path: Path) -> list[Sample]:
    """Load and validate a UTF-8 CSV manifest."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"missing columns: {', '.join(missing)}")
        samples = [Sample.from_row(row) for row in reader]
    validate_manifest(samples)
    return samples


def validate_manifest(samples: Sequence[Sample]) -> None:
    """Reject ambiguous labels, unlicensed rows, duplicates, and split leakage."""

    if not samples:
        raise ManifestError("manifest contains no samples")

    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    group_splits: dict[str, str] = {}

    for sample in samples:
        if not sample.sample_id:
            raise ManifestError("sample_id must not be empty")
        if sample.sample_id in seen_ids:
            raise ManifestError(f"duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)

        if not sample.content_group:
            raise ManifestError(f"{sample.sample_id}: content_group must not be empty")
        if sample.label not in ALLOWED_LABELS:
            raise ManifestError(f"{sample.sample_id}: unsupported label {sample.label!r}")
        if sample.split not in ALLOWED_SPLITS:
            raise ManifestError(f"{sample.sample_id}: unsupported split {sample.split!r}")
        if sample.license_id.lower() in UNKNOWN_LICENSES:
            raise ManifestError(f"{sample.sample_id}: license_id is not approved")
        if len(sample.sha256) != 64 or any(char not in "0123456789abcdef" for char in sample.sha256.lower()):
            raise ManifestError(f"{sample.sample_id}: sha256 must be 64 hexadecimal characters")

        previous_id = seen_hashes.get(sample.sha256.lower())
        if previous_id is not None:
            raise ManifestError(f"duplicate file hash: {previous_id} and {sample.sample_id}")
        seen_hashes[sample.sha256.lower()] = sample.sample_id

        previous_split = group_splits.get(sample.content_group)
        if previous_split is not None and previous_split != sample.split:
            raise ManifestError(
                f"content_group {sample.content_group!r} leaks across {previous_split!r} and {sample.split!r}"
            )
        group_splits[sample.content_group] = sample.split

        if sample.label == "synthetic" and not sample.generator_family:
            raise ManifestError(f"{sample.sample_id}: synthetic rows require generator_family")

    _validate_generator_holdout(samples)


def _validate_generator_holdout(samples: Iterable[Sample]) -> None:
    development_families: set[str] = set()
    holdout_families: set[str] = set()
    for sample in samples:
        if sample.label != "synthetic":
            continue
        family = sample.generator_family.casefold()
        if sample.split == "generator_holdout":
            holdout_families.add(family)
        elif sample.split in {"train", "validation", "calibration"}:
            development_families.add(family)
    overlap = sorted(development_families & holdout_families)
    if overlap:
        raise ManifestError(f"generator families leak into holdout: {', '.join(overlap)}")


def manifest_digest(path: Path) -> str:
    """Return the content hash recorded by model and report artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
