"""Compare evaluation evidence while retaining declared volatile measurements."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPARISON_SCHEMA = "semantic-report-v1"


@dataclass(frozen=True)
class ReportVerification:
    expected_filename: str
    candidate_filename: str
    expected_file_sha256: str
    candidate_file_sha256: str
    comparison_schema: str
    ignored_fields: tuple[str, ...]
    expected_semantic_sha256: str
    candidate_semantic_sha256: str
    semantic_evidence_identical: bool


def compare_evaluation_reports(
    expected_path: Path,
    candidate_path: Path,
    *,
    output: Path,
) -> ReportVerification:
    """Compare two reports after removing only their shared declared fields."""

    expected = _load_report(expected_path)
    candidate = _load_report(candidate_path)
    ignored = _comparison_contract(expected)
    if _comparison_contract(candidate) != ignored:
        raise ValueError("candidate report uses a different reproducibility contract")
    expected_hash = semantic_report_sha256(expected, ignored_fields=ignored)
    candidate_hash = semantic_report_sha256(candidate, ignored_fields=ignored)
    verification = ReportVerification(
        expected_path.name,
        candidate_path.name,
        _sha256(expected_path),
        _sha256(candidate_path),
        COMPARISON_SCHEMA,
        ignored,
        expected_hash,
        candidate_hash,
        expected_hash == candidate_hash,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(verification), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not verification.semantic_evidence_identical:
        raise RuntimeError("candidate evaluation evidence differs from the expected report")
    return verification


def semantic_report_sha256(report: Mapping[str, Any], *, ignored_fields: Sequence[str]) -> str:
    """Hash canonical JSON after applying explicit JSON-pointer-like exclusions."""

    normalized: Any = copy.deepcopy(dict(report))
    for path in ignored_fields:
        segments = _path_segments(path)
        _remove_path(normalized, segments, path)
    rendered = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-verify-report")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    compare_evaluation_reports(options.expected, options.candidate, output=options.output)
    return 0


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation report: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation report must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _comparison_contract(report: Mapping[str, Any]) -> tuple[str, ...]:
    contract = report.get("reproducibility")
    if not isinstance(contract, Mapping) or contract.get("comparison_schema") != COMPARISON_SCHEMA:
        raise ValueError(f"report must declare the {COMPARISON_SCHEMA} comparison schema")
    ignored = contract.get("ignored_fields")
    if not isinstance(ignored, list) or any(not isinstance(path, str) for path in ignored):
        raise ValueError("report ignored_fields must be a list of strings")
    result = tuple(ignored)
    if result != tuple(sorted(set(result))):
        raise ValueError("report ignored_fields must be unique and sorted")
    for path in result:
        _path_segments(path)
    return result


def _path_segments(path: str) -> tuple[str, ...]:
    if not path.startswith("/") or path == "/":
        raise ValueError(f"invalid ignored field path: {path}")
    segments = tuple(path[1:].split("/"))
    if any(not segment or "~" in segment for segment in segments):
        raise ValueError(f"invalid ignored field path: {path}")
    return segments


def _remove_path(value: Any, segments: tuple[str, ...], original: str) -> None:
    head, *tail = segments
    if head == "*":
        if not isinstance(value, list):
            raise ValueError(f"ignored field wildcard does not address a list: {original}")
        for item in value:
            _remove_path(item, tuple(tail), original)
        return
    if not isinstance(value, dict) or head not in value:
        raise ValueError(f"ignored field is absent from report: {original}")
    if tail:
        _remove_path(value[head], tuple(tail), original)
    else:
        del value[head]


if __name__ == "__main__":
    raise SystemExit(main())
