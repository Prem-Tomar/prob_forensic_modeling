"""Offline-friendly construction of a scoped installable library wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SOURCE_DATE_EPOCH = 315532800


@dataclass(frozen=True)
class ReleaseArtifact:
    filename: str
    sha256: str
    size_bytes: int
    source_date_epoch: int
    build_requirements_sha256: str | None
    neural_requirements_sha256: str | None


def build_release_wheel(
    source_root: Path,
    output_directory: Path,
    *,
    manifest: Path | None = None,
) -> ReleaseArtifact:
    """Stage only package inputs, build without isolation, and hash the wheel."""

    source_root = source_root.resolve()
    required = (source_root / "pyproject.toml", source_root / "README.md", source_root / "src")
    if any(not path.exists() for path in required):
        raise ValueError("source root lacks pyproject.toml, README.md, or src")
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forensic-release-") as directory:
        staged = Path(directory)
        wheel_output = staged / "dist"
        shutil.copy2(source_root / "pyproject.toml", staged / "pyproject.toml")
        shutil.copy2(source_root / "README.md", staged / "README.md")
        shutil.copytree(
            source_root / "src",
            staged / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        subprocess.run(
            (
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_output),
                str(staged),
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH)},
        )
        created = tuple(wheel_output.glob("*.whl"))
        if len(created) != 1:
            raise RuntimeError("release build must produce exactly one wheel")
        wheel = output_directory / created[0].name
        shutil.copy2(created[0], wheel)
    artifact = ReleaseArtifact(
        wheel.name,
        _sha256(wheel),
        wheel.stat().st_size,
        SOURCE_DATE_EPOCH,
        _optional_sha256(source_root / "requirements-build.lock"),
        _optional_sha256(source_root / "requirements-neural.lock"),
    )
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-build-release")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    options = parser.parse_args(arguments)
    artifact = build_release_wheel(
        options.source_root,
        options.output_directory,
        manifest=options.manifest,
    )
    print(json.dumps(asdict(artifact), sort_keys=True))
    return 0


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
