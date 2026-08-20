import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from forensic_model.provenance import ProvenanceEvidence, ProvenanceStatus
from forensic_model.provenance_conformance import (
    load_conformance_manifest,
    run_provenance_conformance,
)


SCENARIOS = {
    "signed": "valid",
    "tampered": "invalid",
    "revoked": "invalid",
    "unsupported": "unsupported",
    "absent": "absent",
}


class FixtureVerifier:
    def verify(self, media_path: Path) -> ProvenanceEvidence:
        scenario = media_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(scenario.encode()).hexdigest()
        status = ProvenanceStatus(SCENARIOS[scenario])
        return ProvenanceEvidence(
            status,
            digest,
            digest if status == ProvenanceStatus.VALID else None,
            "trusted-publisher" if status == ProvenanceStatus.VALID else None,
            True if status == ProvenanceStatus.VALID else None,
        )


def write_suite(root: Path, *, scenarios: dict[str, str] = SCENARIOS) -> Path:
    fixtures = []
    for scenario, expected_status in scenarios.items():
        filename = f"{scenario}.bin"
        payload = scenario.encode()
        (root / filename).write_bytes(payload)
        fixtures.append(
            {
                "id": f"fixture-{scenario}",
                "media_path": filename,
                "media_sha256": hashlib.sha256(payload).hexdigest(),
                "scenario": scenario,
                "expected_status": expected_status,
                "expected_generator_asserted": True if scenario == "signed" else None,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "provenance-conformance-v1",
                "attribution": {
                    "source_name": "Public fixture suite",
                    "source_url": "https://example.test/fixtures",
                    "license": "CC0-1.0",
                    "citation": "Public fixture suite, version 1",
                },
                "fixtures": fixtures,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class ProvenanceConformanceTests(unittest.TestCase):
    def test_passes_all_required_scenarios_and_writes_path_free_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_suite(root)
            output = root / "report.json"
            report = run_provenance_conformance(manifest, root, FixtureVerifier(), output=output)
            rendered = output.read_text(encoding="utf-8")

        self.assertTrue(report["acceptance"]["all_required_scenarios_passed"])
        self.assertEqual(report["fixture_count"], 5)
        self.assertNotIn(directory, rendered)
        self.assertNotIn("media_path", rendered)

    def test_rejects_missing_scenario_and_contradictory_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = dict(SCENARIOS)
            del missing["revoked"]
            with self.assertRaisesRegex(ValueError, "missing required scenarios: revoked"):
                load_conformance_manifest(write_suite(root, scenarios=missing), media_root=root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_suite(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["fixtures"][0]["expected_status"] = "absent"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                load_conformance_manifest(manifest, media_root=root)

    def test_rejects_tampered_bytes_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_suite(root)
            (root / "signed.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                load_conformance_manifest(manifest, media_root=root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_suite(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["fixtures"][0]["media_path"] = "../outside.bin"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must remain under"):
                load_conformance_manifest(manifest, media_root=root)

    def test_writes_failure_evidence_before_raising(self) -> None:
        class WrongVerifier(FixtureVerifier):
            def verify(self, media_path: Path) -> ProvenanceEvidence:
                evidence = super().verify(media_path)
                if media_path.read_text(encoding="utf-8") == "revoked":
                    return ProvenanceEvidence(ProvenanceStatus.INDETERMINATE, evidence.media_sha256)
                return evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_suite(root)
            output = root / "report.json"
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                run_provenance_conformance(manifest, root, WrongVerifier(), output=output)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(report["acceptance"]["scenario_results"]["revoked"])
        self.assertFalse(report["acceptance"]["all_required_scenarios_passed"])


if __name__ == "__main__":
    unittest.main()
