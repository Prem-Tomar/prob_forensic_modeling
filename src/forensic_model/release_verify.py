"""Verify a built wheel in an isolated dependency-free Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import venv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ReleaseVerification:
    wheel_filename: str
    wheel_sha256: str
    expected_smoke_sha256: str
    installed_smoke_sha256: str
    smoke_report_byte_identical: bool
    install_mode: str = "local_wheel_no_index_no_dependencies"


def verify_release_wheel(
    wheel: Path,
    expected_smoke_report: Path,
    *,
    output: Path,
) -> ReleaseVerification:
    """Install only the local wheel, run smoke evaluation, and compare exact bytes."""

    if not wheel.is_file() or not expected_smoke_report.is_file():
        raise ValueError("verification requires an existing wheel and expected smoke report")
    with tempfile.TemporaryDirectory(prefix="forensic-wheel-verify-") as directory:
        root = Path(directory)
        environment = root / "venv"
        installed_report = root / "smoke.json"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel.resolve()),
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            (
                str(python),
                "-m",
                "forensic_model.cli",
                "smoke-evaluate",
                "--output",
                str(installed_report),
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        installed_bytes = installed_report.read_bytes()
    expected_bytes = expected_smoke_report.read_bytes()
    verification = ReleaseVerification(
        wheel.name,
        _sha256(wheel),
        hashlib.sha256(expected_bytes).hexdigest(),
        hashlib.sha256(installed_bytes).hexdigest(),
        installed_bytes == expected_bytes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(verification), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not verification.smoke_report_byte_identical:
        raise RuntimeError("installed wheel smoke report differs from the expected report")
    return verification


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-verify-release")
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    verify_release_wheel(
        options.wheel,
        options.expected_smoke_report,
        output=options.output,
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
