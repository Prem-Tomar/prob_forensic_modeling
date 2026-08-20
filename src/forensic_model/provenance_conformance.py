"""Run a local provenance verifier against an attributed conformance suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from forensic_model.provenance import ProvenanceStatus, ProvenanceVerifier, check_provenance
from forensic_model.provenance_process import JsonProcessVerifier, SignerTrustPolicy


SCHEMA = "provenance-conformance-v1"
REQUIRED_SCENARIOS = ("absent", "revoked", "signed", "tampered", "unsupported")
EXPECTED_SCENARIO_STATUS = {
    "absent": ProvenanceStatus.ABSENT,
    "revoked": ProvenanceStatus.INVALID,
    "signed": ProvenanceStatus.VALID,
    "tampered": ProvenanceStatus.INVALID,
    "unsupported": ProvenanceStatus.UNSUPPORTED,
}


@dataclass(frozen=True)
class ConformanceAttribution:
    source_name: str
    source_url: str
    license: str
    citation: str


@dataclass(frozen=True)
class ConformanceFixture:
    fixture_id: str
    media_path: Path
    media_sha256: str
    scenario: str
    expected_status: ProvenanceStatus
    expected_generator_asserted: bool | None


@dataclass(frozen=True)
class FixtureResult:
    fixture_id: str
    scenario: str
    media_sha256: str
    expected_status: str
    observed_status: str
    expected_generator_asserted: bool | None
    observed_generator_asserted: bool | None
    signer: str | None
    passed: bool


def run_provenance_conformance(
    manifest: Path,
    media_root: Path,
    verifier: ProvenanceVerifier,
    *,
    output: Path,
) -> dict[str, object]:
    """Verify every fixture and write a path-free, machine-readable gate report."""

    attribution, fixtures = load_conformance_manifest(manifest, media_root=media_root)
    results: list[FixtureResult] = []
    for fixture in fixtures:
        evidence = check_provenance(fixture.media_path, verifier)
        assertion_matches = (
            fixture.expected_generator_asserted is None
            or evidence.generator_asserted is fixture.expected_generator_asserted
        )
        results.append(
            FixtureResult(
                fixture.fixture_id,
                fixture.scenario,
                fixture.media_sha256,
                fixture.expected_status.value,
                evidence.status.value,
                fixture.expected_generator_asserted,
                evidence.generator_asserted,
                evidence.signer,
                evidence.status == fixture.expected_status and assertion_matches,
            )
        )
    scenario_results = {
        scenario: all(result.passed for result in results if result.scenario == scenario)
        for scenario in REQUIRED_SCENARIOS
    }
    report: dict[str, object] = {
        "claim_scope": "local_verifier_conformance_not_cryptographic_verification_by_this_library",
        "schema": SCHEMA,
        "manifest_sha256": _sha256(manifest),
        "attribution": asdict(attribution),
        "fixture_count": len(results),
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "fixtures": [asdict(result) for result in results],
        "acceptance": {
            "scenario_results": scenario_results,
            "all_fixtures_passed": all(result.passed for result in results),
            "all_required_scenarios_passed": all(scenario_results.values()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(scenario_results.values()):
        raise RuntimeError("provenance verifier did not pass the conformance suite")
    return report


def load_conformance_manifest(
    manifest: Path,
    *,
    media_root: Path,
) -> tuple[ConformanceAttribution, tuple[ConformanceFixture, ...]]:
    """Load, validate, and hash-check a portable fixture manifest."""

    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read conformance manifest: {manifest.name}") from error
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError(f"conformance manifest must use schema {SCHEMA}")
    attribution = _parse_attribution(value.get("attribution"))
    raw_fixtures = value.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("conformance manifest fixtures must be a non-empty list")
    root = media_root.resolve()
    fixtures = tuple(_parse_fixture(item, root=root) for item in raw_fixtures)
    identifiers = [fixture.fixture_id for fixture in fixtures]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("conformance fixture ids must be unique")
    paths = [fixture.media_path for fixture in fixtures]
    if len(paths) != len(set(paths)):
        raise ValueError("each conformance fixture must use distinct media bytes")
    scenarios = {fixture.scenario for fixture in fixtures}
    missing = sorted(set(REQUIRED_SCENARIOS) - scenarios)
    if missing:
        raise ValueError(f"conformance manifest is missing required scenarios: {', '.join(missing)}")
    return attribution, fixtures


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-test-provenance")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--verifier-argument", action="append", default=[])
    parser.add_argument("--trusted-signer", action="append", default=[])
    parser.add_argument("--allow-unlisted-signers", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-output-bytes", type=int, default=1024 * 1024)
    options = parser.parse_args(arguments)
    verifier = JsonProcessVerifier(
        (str(options.verifier), *options.verifier_argument),
        trust_policy=SignerTrustPolicy(
            frozenset(options.trusted_signer),
            allow_unlisted_signers=options.allow_unlisted_signers,
        ),
        timeout_seconds=options.timeout_seconds,
        max_output_bytes=options.max_output_bytes,
    )
    run_provenance_conformance(
        options.manifest,
        options.media_root,
        verifier,
        output=options.output,
    )
    return 0


def _parse_attribution(value: object) -> ConformanceAttribution:
    if not isinstance(value, Mapping):
        raise ValueError("conformance attribution must be an object")
    source_name = _nonempty_string(value.get("source_name"), "attribution source_name")
    source_url = _nonempty_string(value.get("source_url"), "attribution source_url")
    license_name = _nonempty_string(value.get("license"), "attribution license")
    citation = _nonempty_string(value.get("citation"), "attribution citation")
    if not source_url.startswith(("https://", "http://")):
        raise ValueError("conformance source_url must use HTTP(S)")
    return ConformanceAttribution(source_name, source_url, license_name, citation)


def _parse_fixture(value: object, *, root: Path) -> ConformanceFixture:
    if not isinstance(value, Mapping):
        raise ValueError("each conformance fixture must be an object")
    fixture_id = _nonempty_string(value.get("id"), "fixture id")
    relative = Path(_nonempty_string(value.get("media_path"), "fixture media_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"fixture {fixture_id} media_path must remain under media_root")
    media_path = (root / relative).resolve()
    if not media_path.is_relative_to(root) or not media_path.is_file():
        raise ValueError(f"fixture {fixture_id} media_path is missing or escapes media_root")
    media_sha256 = _digest(value.get("media_sha256"))
    if _sha256(media_path) != media_sha256:
        raise ValueError(f"fixture {fixture_id} media SHA-256 does not match")
    scenario = _nonempty_string(value.get("scenario"), "fixture scenario")
    if scenario not in EXPECTED_SCENARIO_STATUS:
        raise ValueError(f"fixture {fixture_id} has an unknown scenario")
    try:
        expected_status = ProvenanceStatus(_nonempty_string(value.get("expected_status"), "expected_status"))
    except ValueError as error:
        raise ValueError(f"fixture {fixture_id} has an unknown expected_status") from error
    if expected_status != EXPECTED_SCENARIO_STATUS[scenario]:
        raise ValueError(f"fixture {fixture_id} expected_status contradicts its scenario")
    asserted = value.get("expected_generator_asserted")
    if asserted is not None and not isinstance(asserted, bool):
        raise ValueError(f"fixture {fixture_id} expected_generator_asserted must be boolean or null")
    return ConformanceFixture(fixture_id, media_path, media_sha256, scenario, expected_status, asserted)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("fixture media_sha256 must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("fixture media_sha256 must be a SHA-256 hex digest") from error
    return value.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
