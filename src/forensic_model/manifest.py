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
from urllib.parse import urlparse


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
    "source_url",
    "source_type",
    "license_id",
    "citation",
    "generator_family",
    "transformation",
    "semantic_category",
    "capture_device",
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
    source_url: str
    source_type: str
    license_id: str
    citation: str
    generator_family: str
    transformation: str
    semantic_category: str = ""
    capture_device: str = ""

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
    source_attributions: dict[str, tuple[str, str, str]] = {}

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
        if not sample.source or not sample.source_type or not sample.transformation:
            raise ManifestError(f"{sample.sample_id}: source, source_type, and transformation are required")
        if not sample.semantic_category:
            raise ManifestError(f"{sample.sample_id}: semantic_category must not be empty")
        if sample.source_type.casefold() in {"camera", "captured"} and not sample.capture_device:
            raise ManifestError(f"{sample.sample_id}: camera rows require capture_device")
        parsed_url = urlparse(sample.source_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ManifestError(f"{sample.sample_id}: source_url must be an HTTP(S) URL")
        if not sample.citation:
            raise ManifestError(f"{sample.sample_id}: citation must not be empty")
        attribution = (sample.source_url, sample.license_id, sample.citation)
        previous_attribution = source_attributions.get(sample.source)
        if previous_attribution is not None and previous_attribution != attribution:
            raise ManifestError(f"{sample.sample_id}: source attribution is inconsistent")
        source_attributions[sample.source] = attribution
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
    _validate_source_holdout(samples)


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


def _validate_source_holdout(samples: Iterable[Sample]) -> None:
    development_sources: set[str] = set()
    development_devices: set[str] = set()
    holdout_rows: list[Sample] = []
    for sample in samples:
        if sample.label == "camera_or_human" and sample.split in {"train", "validation", "calibration"}:
            development_sources.add(sample.source.casefold())
            if sample.capture_device:
                development_devices.add(sample.capture_device.casefold())
        elif sample.split == "source_holdout":
            if sample.label != "camera_or_human":
                raise ManifestError(f"{sample.sample_id}: source_holdout rows must be camera_or_human")
            holdout_rows.append(sample)
    for sample in holdout_rows:
        source_seen = sample.source.casefold() in development_sources
        device_seen = bool(sample.capture_device) and sample.capture_device.casefold() in development_devices
        if source_seen and (not sample.capture_device or device_seen):
            raise ManifestError(
                f"{sample.sample_id}: source_holdout must introduce an unseen source or capture_device"
            )


def manifest_digest(path: Path) -> str:
    """Return the content hash recorded by model and report artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
