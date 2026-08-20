"""Portable, hash-verified image corpora backed by governed manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from forensic_model.data_audit import sha256_file
from forensic_model.manifest import Sample, load_manifest, manifest_digest
from forensic_model.neural_data import ImageExample


@dataclass(frozen=True)
class ManifestSourceAttribution:
    source: str
    source_url: str
    license_id: str
    citation: str
    rows: int


@dataclass(frozen=True)
class ManifestImageAudit:
    manifest_sha256: str
    verified_files: int
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, int]
    generator_counts: Mapping[str, int]
    source_license_counts: Mapping[str, int]
    attributions: tuple[ManifestSourceAttribution, ...]


@dataclass(frozen=True)
class MatchedGeneratorHoldout:
    generator_family: str
    real_controls: tuple[ImageExample, ...]
    synthetic_examples: tuple[ImageExample, ...]

    @property
    def examples(self) -> tuple[ImageExample, ...]:
        return self.real_controls + self.synthetic_examples


@dataclass(frozen=True)
class ManifestImageCorpus:
    splits: Mapping[str, tuple[ImageExample, ...]]
    audit: ManifestImageAudit

    def examples(self, split: str) -> tuple[ImageExample, ...]:
        """Return one frozen split, rejecting accidental name fallbacks."""

        try:
            return self.splits[split]
        except KeyError as error:
            raise ValueError(f"manifest does not contain split: {split}") from error

    def generator_holdouts(self) -> Mapping[str, tuple[ImageExample, ...]]:
        """Group synthetic generator-holdout rows by normalized family name."""

        grouped: dict[str, list[ImageExample]] = {}
        for example in self.splits.get("generator_holdout", ()):
            if example.label == 1:
                grouped.setdefault(example.generator_family.casefold(), []).append(example)
        return MappingProxyType({family: tuple(rows) for family, rows in sorted(grouped.items())})

    def matched_generator_holdouts(self) -> Mapping[str, MatchedGeneratorHoldout]:
        """Require a one-to-one real control for each synthetic content identity."""

        holdout = self.splits.get("generator_holdout", ())
        real_by_group: dict[str, list[ImageExample]] = {}
        synthetic_by_family_group: dict[str, dict[str, list[ImageExample]]] = {}
        for example in holdout:
            if example.label == 0:
                real_by_group.setdefault(example.content_group, []).append(example)
            else:
                family = example.generator_family.casefold()
                synthetic_by_family_group.setdefault(family, {}).setdefault(example.content_group, []).append(example)
        matched: dict[str, MatchedGeneratorHoldout] = {}
        for family, by_group in sorted(synthetic_by_family_group.items()):
            real_controls = []
            synthetic_examples = []
            for group, generated in sorted(by_group.items()):
                controls = real_by_group.get(group, [])
                if len(controls) != 1 or len(generated) != 1:
                    raise ValueError(
                        f"generator holdout {family!r} requires one real and one synthetic row for content group {group!r}"
                    )
                _validate_matched_slice_metadata(family, group, controls[0], generated[0])
                real_controls.append(controls[0])
                synthetic_examples.append(generated[0])
            matched[family] = MatchedGeneratorHoldout(
                family,
                tuple(real_controls),
                tuple(synthetic_examples),
            )
        return MappingProxyType(matched)


def load_manifest_image_corpus(manifest_path: Path, *, media_root: Path | None = None) -> ManifestImageCorpus:
    """Resolve portable paths and verify every manifest hash before use."""

    manifest_path = manifest_path.resolve()
    root = (media_root or manifest_path.parent).resolve()
    samples = load_manifest(manifest_path)
    by_split: dict[str, list[ImageExample]] = {}
    split_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    source_license_counts: dict[str, int] = {}
    source_attributions: dict[str, tuple[str, str, str]] = {}
    for sample in samples:
        path = _resolve_media_path(root, sample)
        if not path.is_file():
            raise ValueError(f"manifest media is missing: {sample.sample_id}")
        if sha256_file(path) != sample.sha256.lower():
            raise ValueError(f"manifest media hash mismatch: {sample.sample_id}")
        example = ImageExample(
            path=path,
            label=int(sample.label == "synthetic"),
            content_group=sample.content_group,
            source=sample.source,
            generator_family=sample.generator_family,
            semantic_category=sample.semantic_category,
            capture_device=sample.capture_device,
        )
        by_split.setdefault(sample.split, []).append(example)
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
        if sample.generator_family:
            family = sample.generator_family.casefold()
            generator_counts[family] = generator_counts.get(family, 0) + 1
        source_license = f"{sample.source} ({sample.license_id})"
        source_license_counts[source_license] = source_license_counts.get(source_license, 0) + 1
        source_attributions[sample.source] = (sample.source_url, sample.license_id, sample.citation)
    frozen_splits = MappingProxyType(
        {
            split: tuple(sorted(rows, key=lambda row: (row.content_group, row.label, row.source)))
            for split, rows in sorted(by_split.items())
        }
    )
    audit = ManifestImageAudit(
        manifest_digest(manifest_path),
        len(samples),
        MappingProxyType(dict(sorted(split_counts.items()))),
        MappingProxyType(dict(sorted(label_counts.items()))),
        MappingProxyType(dict(sorted(generator_counts.items()))),
        MappingProxyType(dict(sorted(source_license_counts.items()))),
        tuple(
            ManifestSourceAttribution(
                source,
                source_url,
                license_id,
                citation,
                source_license_counts[f"{source} ({license_id})"],
            )
            for source, (source_url, license_id, citation) in sorted(source_attributions.items())
        ),
    )
    return ManifestImageCorpus(frozen_splits, audit)


def _resolve_media_path(root: Path, sample: Sample) -> Path:
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
    control: ImageExample,
    generated: ImageExample,
) -> None:
    for field in ("semantic_category", "capture_device"):
        control_value = getattr(control, field)
        generated_value = getattr(generated, field)
        if control_value and generated_value and control_value.casefold() != generated_value.casefold():
            raise ValueError(
                f"generator holdout {family!r} has conflicting {field} values for content group {group!r}"
            )
