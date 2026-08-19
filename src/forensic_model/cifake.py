"""Pillow-only CIFAKE discovery shared by audit and neural experiments."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from forensic_model.data_audit import CandidateSample, sha256_file


def discover_cifake_candidates(root: Path) -> tuple[CandidateSample, ...]:
    """Return auditable CIFAKE candidates before duplicate removal or splitting."""

    candidates = []
    for path in sorted(root.glob("*/*/*.jpg")):
        relative = path.relative_to(root).as_posix()
        label = "synthetic" if path.parent.name.casefold() == "fake" else "camera_or_human"
        candidates.append(
            CandidateSample(
                sample_id=relative,
                path=path,
                label=label,
                sha256=sha256_file(path),
                perceptual_hash=difference_hash(path),
            )
        )
    return tuple(candidates)


def difference_hash(path: Path) -> str:
    """Return a deterministic 64-bit grayscale difference hash."""

    with Image.open(path) as source:
        values = list(source.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (values[row * 9 + column] > values[row * 9 + column + 1])
    return f"{bits:016x}"
