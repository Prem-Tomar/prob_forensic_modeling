"""Reproducible evidence for manual review of perceptual-hash collisions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from forensic_model.cifake import discover_cifake_candidates
from forensic_model.data_audit import CandidateSample


@dataclass(frozen=True)
class CollisionComparison:
    first_sha256: str
    second_sha256: str
    labels_match: bool
    dimensions_match: bool
    grayscale_mean_absolute_error: float
    rgb_mean_absolute_error: float
    automated_screen: str


def review_perceptual_collisions(
    candidates: Sequence[CandidateSample],
    *,
    thumbnail_size: int = 64,
    low_distance_threshold: float = 0.02,
) -> dict[str, object]:
    """Measure compact-hash collisions without silently changing identities."""

    if thumbnail_size < 8:
        raise ValueError("thumbnail size must be at least 8")
    if not 0.0 < low_distance_threshold < 1.0:
        raise ValueError("low distance threshold must be within (0, 1)")
    by_perceptual_hash: dict[str, dict[str, CandidateSample]] = {}
    for candidate in candidates:
        if candidate.perceptual_hash:
            by_perceptual_hash.setdefault(candidate.perceptual_hash.lower(), {})[
                candidate.sha256.lower()
            ] = candidate
    collision_groups = [
        (perceptual_hash, tuple(rows.values()))
        for perceptual_hash, rows in sorted(by_perceptual_hash.items())
        if len(rows) > 1
    ]
    records = []
    total_pairs = 0
    low_distance_pairs = 0
    cross_label_pairs = 0
    for perceptual_hash, members in collision_groups:
        comparisons = []
        for first, second in itertools.combinations(sorted(members, key=lambda row: row.sha256), 2):
            comparison = _compare(
                first,
                second,
                thumbnail_size=thumbnail_size,
                low_distance_threshold=low_distance_threshold,
            )
            comparisons.append(asdict(comparison))
            total_pairs += 1
            low_distance_pairs += comparison.automated_screen == "low_pixel_distance"
            cross_label_pairs += not comparison.labels_match
        records.append(
            {
                "group_id": hashlib.sha256(perceptual_hash.encode("ascii")).hexdigest(),
                "member_count": len(members),
                "members": [
                    {
                        "sha256": member.sha256.lower(),
                        "label": member.label,
                    }
                    for member in sorted(members, key=lambda row: row.sha256)
                ],
                "comparisons": comparisons,
            }
        )
    return {
        "claim_scope": "perceptual_hash_collision_review_not_automatic_deduplication",
        "configuration": {
            "thumbnail_size": thumbnail_size,
            "low_distance_threshold": low_distance_threshold,
            "distance_normalization": "mean_absolute_error_divided_by_255",
        },
        "candidate_count": len(candidates),
        "collision_group_count": len(collision_groups),
        "pair_comparison_count": total_pairs,
        "low_pixel_distance_pair_count": low_distance_pairs,
        "cross_label_pair_count": cross_label_pairs,
        "groups": records,
        "policy": (
            "Compact-hash collisions remain separate identities. The automated screen prioritizes visual review "
            "but never merges, removes, or reassigns a sample."
        ),
    }


def write_contact_sheet(
    candidates: Sequence[CandidateSample],
    report: dict[str, object],
    *,
    output: Path,
    cell_size: int = 128,
) -> None:
    """Render local-only review thumbnails identified by digest prefixes."""

    if cell_size < 64:
        raise ValueError("contact-sheet cells must be at least 64 pixels")
    by_sha = {candidate.sha256.lower(): candidate for candidate in candidates}
    groups = report["groups"]
    if not isinstance(groups, list):
        raise ValueError("report groups must be a list")
    member_counts = [int(group["member_count"]) for group in groups]
    columns = max(member_counts, default=1)
    row_height = cell_size + 36
    canvas = Image.new("RGB", (columns * cell_size, max(1, len(groups)) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, group in enumerate(groups):
        for column_index, member in enumerate(group["members"]):
            candidate = by_sha[member["sha256"]]
            with Image.open(candidate.path) as source:
                thumbnail = source.convert("RGB")
            thumbnail.thumbnail((cell_size - 8, cell_size - 8), Image.Resampling.LANCZOS)
            left = column_index * cell_size + (cell_size - thumbnail.width) // 2
            top = row_index * row_height + (cell_size - thumbnail.height) // 2
            canvas.paste(thumbnail, (left, top))
            draw.text(
                (column_index * cell_size + 4, row_index * row_height + cell_size),
                f"{member['label'][:4]} {member['sha256'][:10]}",
                fill="black",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "PNG")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-review-image-collisions")
    parser.add_argument("--cifake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument("--low-distance-threshold", type=float, default=0.02)
    options = parser.parse_args(arguments)
    candidates = discover_cifake_candidates(options.cifake_root)
    report = review_perceptual_collisions(
        candidates,
        thumbnail_size=options.thumbnail_size,
        low_distance_threshold=options.low_distance_threshold,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if options.contact_sheet is not None:
        write_contact_sheet(candidates, report, output=options.contact_sheet)
    return 0


def _compare(
    first: CandidateSample,
    second: CandidateSample,
    *,
    thumbnail_size: int,
    low_distance_threshold: float,
) -> CollisionComparison:
    first_rgb, first_dimensions = _normalized_pixels(first.path, thumbnail_size, "RGB")
    second_rgb, second_dimensions = _normalized_pixels(second.path, thumbnail_size, "RGB")
    first_gray, _ = _normalized_pixels(first.path, thumbnail_size, "L")
    second_gray, _ = _normalized_pixels(second.path, thumbnail_size, "L")
    rgb_error = _mean_absolute_error(first_rgb, second_rgb)
    gray_error = _mean_absolute_error(first_gray, second_gray)
    return CollisionComparison(
        first.sha256.lower(),
        second.sha256.lower(),
        first.label == second.label,
        first_dimensions == second_dimensions,
        gray_error,
        rgb_error,
        "low_pixel_distance" if gray_error <= low_distance_threshold else "hash_collision_only",
    )


def _normalized_pixels(path: Path, size: int, mode: str) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as source:
        dimensions = source.size
        normalized = source.convert(mode).resize((size, size), Image.Resampling.BILINEAR)
    return normalized.tobytes(), dimensions


def _mean_absolute_error(first: bytes, second: bytes) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("pixel vectors must be aligned and non-empty")
    return sum(abs(left - right) for left, right in zip(first, second)) / (255.0 * len(first))


if __name__ == "__main__":
    raise SystemExit(main())
