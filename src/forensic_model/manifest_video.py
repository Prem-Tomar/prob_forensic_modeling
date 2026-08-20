"""Portable, hash-verified video corpora backed by governed manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from forensic_model.data_audit import sha256_file
from forensic_model.manifest import Sample, load_manifest, manifest_digest
from forensic_model.neural_video_data import VideoExample


VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm"})


@dataclass(frozen=True)
class ManifestVideoAttribution:
    source: str
    source_url: str
    license_id: str
    citation: str
    rows: int


@dataclass(frozen=True)
class ManifestVideoAudit:
    manifest_sha256: str
    verified_files: int
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, int]
    generator_counts: Mapping[str, int]
    attributions: tuple[ManifestVideoAttribution, ...]


@dataclass(frozen=True)
class MatchedVideoHoldout:
    generator_family: str
    real_controls: tuple[VideoExample, ...]
    synthetic_examples: tuple[VideoExample, ...]

    @property
    def examples(self) -> tuple[VideoExample, ...]:
        return self.real_controls + self.synthetic_examples


@dataclass(frozen=True)
class ManifestVideoCorpus:
    splits: Mapping[str, tuple[VideoExample, ...]]
    audit: ManifestVideoAudit

    def examples(self, split: str) -> tuple[VideoExample, ...]:
        try:
            return self.splits[split]
        except KeyError as error:
            raise ValueError(f"manifest does not contain split: {split}") from error

    def matched_generator_holdouts(self) -> Mapping[str, MatchedVideoHoldout]:
        holdout = self.splits.get("generator_holdout", ())
        real_by_group: dict[str, list[VideoExample]] = {}
        synthetic_by_family_group: dict[str, dict[str, list[VideoExample]]] = {}
        for example in holdout:
            if example.label == 0:
                real_by_group.setdefault(example.content_group, []).append(example)
            else:
                family = example.generator_family.casefold()
                synthetic_by_family_group.setdefault(family, {}).setdefault(example.content_group, []).append(example)
        matched: dict[str, MatchedVideoHoldout] = {}
        for family, by_group in sorted(synthetic_by_family_group.items()):
            controls = []
            generated = []
            for group, synthetic_rows in sorted(by_group.items()):
                real_rows = real_by_group.get(group, [])
                if len(real_rows) != 1 or len(synthetic_rows) != 1:
                    raise ValueError(
                        f"video holdout {family!r} requires one real and one synthetic clip for content group {group!r}"
                    )
                _validate_matched_slice_metadata(family, group, real_rows[0], synthetic_rows[0])
                controls.append(real_rows[0])
                generated.append(synthetic_rows[0])
            matched[family] = MatchedVideoHoldout(family, tuple(controls), tuple(generated))
        return MappingProxyType(matched)


def load_manifest_video_corpus(manifest_path: Path, *, media_root: Path | None = None) -> ManifestVideoCorpus:
    """Resolve portable clip paths and verify exact bytes before decoding."""

    manifest_path = manifest_path.resolve()
    root = (media_root or manifest_path.parent).resolve()
    samples = load_manifest(manifest_path)
    by_split: dict[str, list[VideoExample]] = {}
    split_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    attribution_rows: dict[str, int] = {}
    attribution_values: dict[str, tuple[str, str, str]] = {}
    for sample in samples:
        path = _resolve_video_path(root, sample)
        if not path.is_file():
            raise ValueError(f"manifest video is missing: {sample.sample_id}")
        if path.suffix.casefold() not in VIDEO_SUFFIXES:
            raise ValueError(f"manifest row is not a supported video: {sample.sample_id}")
        if sha256_file(path) != sample.sha256.lower():
            raise ValueError(f"manifest video hash mismatch: {sample.sample_id}")
        example = VideoExample(
            path,
            int(sample.label == "synthetic"),
            sample.content_group,
            sample.source,
            sample.generator_family,
            sample.split,
            "video",
            sample.semantic_category,
            sample.capture_device,
        )
        by_split.setdefault(sample.split, []).append(example)
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
        if sample.generator_family:
            family = sample.generator_family.casefold()
            generator_counts[family] = generator_counts.get(family, 0) + 1
        attribution_rows[sample.source] = attribution_rows.get(sample.source, 0) + 1
        attribution_values[sample.source] = (sample.source_url, sample.license_id, sample.citation)
    splits = MappingProxyType(
        {
            split: tuple(sorted(rows, key=lambda row: (row.content_group, row.label, row.source)))
            for split, rows in sorted(by_split.items())
        }
    )
    audit = ManifestVideoAudit(
        manifest_digest(manifest_path),
        len(samples),
        MappingProxyType(dict(sorted(split_counts.items()))),
        MappingProxyType(dict(sorted(label_counts.items()))),
        MappingProxyType(dict(sorted(generator_counts.items()))),
        tuple(
            ManifestVideoAttribution(source, source_url, license_id, citation, attribution_rows[source])
            for source, (source_url, license_id, citation) in sorted(attribution_values.items())
        ),
    )
    return ManifestVideoCorpus(splits, audit)


def _resolve_video_path(root: Path, sample: Sample) -> Path:
    relative = Path(sample.path)
    if relative.is_absolute():
        raise ValueError(f"manifest path must be relative: {sample.sample_id}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes media root: {sample.sample_id}") from error
    return resolved


def _validate_matched_slice_metadata(
    family: str,
    group: str,
    control: VideoExample,
    generated: VideoExample,
) -> None:
    if control.semantic_category.casefold() != generated.semantic_category.casefold():
        raise ValueError(
            f"video holdout {family!r} has conflicting semantic_category values for content group {group!r}"
        )
    if control.capture_device and generated.capture_device:
        if control.capture_device.casefold() != generated.capture_device.casefold():
            raise ValueError(
                f"video holdout {family!r} has conflicting capture_device values for content group {group!r}"
            )
