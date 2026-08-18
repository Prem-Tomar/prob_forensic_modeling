import hashlib
import tempfile
import unittest
from pathlib import Path

from forensic_model.decision import DetectionResult
from forensic_model.provenance import (
    ProvenanceEvidence,
    ProvenanceStatus,
    check_provenance,
    fuse_evidence,
)


def pixel(decision: str) -> DetectionResult:
    return DetectionResult(
        decision=decision,
        probability_synthetic=0.9 if decision == "synthetic" else 0.1,
        decision_confidence=0.9,
        raw_score=2.0 if decision == "synthetic" else -2.0,
        threshold=0.5,
        abstained=decision == "abstain",
        calibration="platt",
        reasons=(),
        contributions={},
    )


class StaticVerifier:
    def __init__(self, evidence: ProvenanceEvidence) -> None:
        self.evidence = evidence

    def verify(self, media_path: Path) -> ProvenanceEvidence:
        return self.evidence


class ProvenanceTests(unittest.TestCase):
    def test_absence_is_reported_without_synthetic_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            evidence = check_provenance(path)
        self.assertEqual(evidence.status, ProvenanceStatus.ABSENT)
        self.assertEqual(fuse_evidence(pixel("camera_or_human"), evidence).decision, "camera_or_human")

    def test_mismatched_verifier_binding_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            claimed = ProvenanceEvidence(ProvenanceStatus.VALID, "0" * 64, "0" * 64, generator_asserted=True)
            evidence = check_provenance(path, StaticVerifier(claimed))
        self.assertEqual(evidence.status, ProvenanceStatus.INVALID)

    def test_verified_agreement_is_preserved(self) -> None:
        digest = hashlib.sha256(b"media").hexdigest()
        evidence = ProvenanceEvidence(ProvenanceStatus.VALID, digest, digest, "fixture-signer", True)
        result = fuse_evidence(pixel("synthetic"), evidence)
        self.assertEqual(result.decision, "synthetic")
        self.assertFalse(result.conflict)

    def test_verified_conflict_abstains(self) -> None:
        digest = hashlib.sha256(b"media").hexdigest()
        evidence = ProvenanceEvidence(ProvenanceStatus.VALID, digest, digest, "fixture-signer", False)
        result = fuse_evidence(pixel("synthetic"), evidence)
        self.assertEqual(result.decision, "abstain")
        self.assertTrue(result.conflict)

    def test_invalid_credentials_do_not_become_synthetic_label(self) -> None:
        evidence = ProvenanceEvidence(ProvenanceStatus.INVALID, "a" * 64)
        result = fuse_evidence(pixel("camera_or_human"), evidence)
        self.assertEqual(result.decision, "abstain")

    def test_non_verifying_states_leave_pixel_evidence_visible(self) -> None:
        for status in (ProvenanceStatus.UNSUPPORTED, ProvenanceStatus.INDETERMINATE):
            with self.subTest(status=status):
                evidence = ProvenanceEvidence(status, "a" * 64)
                result = fuse_evidence(pixel("synthetic"), evidence)
                self.assertEqual(result.decision, "synthetic")
                self.assertFalse(result.conflict)


if __name__ == "__main__":
    unittest.main()
