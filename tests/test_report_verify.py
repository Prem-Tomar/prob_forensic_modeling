import json
import tempfile
import unittest
from pathlib import Path

from forensic_model.report_verify import compare_evaluation_reports, semantic_report_sha256


class ReportVerificationTests(unittest.TestCase):
    def _report(self, *, elapsed: float, metric: float = 0.8) -> dict[str, object]:
        return {
            "metric": metric,
            "training_history": [{"epoch": 1, "elapsed_seconds": elapsed, "loss": 0.2}],
            "reproducibility": {
                "comparison_schema": "semantic-report-v1",
                "ignored_fields": ["/training_history/*/elapsed_seconds"],
            },
        }

    def test_ignores_only_declared_volatile_fields(self) -> None:
        ignored = ("/training_history/*/elapsed_seconds",)
        self.assertEqual(
            semantic_report_sha256(self._report(elapsed=1.0), ignored_fields=ignored),
            semantic_report_sha256(self._report(elapsed=9.0), ignored_fields=ignored),
        )
        self.assertNotEqual(
            semantic_report_sha256(self._report(elapsed=1.0), ignored_fields=ignored),
            semantic_report_sha256(self._report(elapsed=1.0, metric=0.7), ignored_fields=ignored),
        )

    def test_writes_successful_semantic_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected.json"
            candidate = root / "candidate.json"
            output = root / "verification.json"
            expected.write_text(json.dumps(self._report(elapsed=1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(self._report(elapsed=4.0)), encoding="utf-8")

            result = compare_evaluation_reports(expected, candidate, output=output)
            recorded = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(result.semantic_evidence_identical)
        self.assertTrue(recorded["semantic_evidence_identical"])
        self.assertEqual(recorded["expected_semantic_sha256"], recorded["candidate_semantic_sha256"])

    def test_rejects_different_contracts_or_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected.json"
            candidate = root / "candidate.json"
            output = root / "verification.json"
            expected.write_text(json.dumps(self._report(elapsed=1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(self._report(elapsed=1.0, metric=0.7)), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "evidence differs"):
                compare_evaluation_reports(expected, candidate, output=output)

            changed_contract = self._report(elapsed=1.0)
            changed_contract["reproducibility"] = {
                "comparison_schema": "semantic-report-v1",
                "ignored_fields": [],
            }
            candidate.write_text(json.dumps(changed_contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different reproducibility contract"):
                compare_evaluation_reports(expected, candidate, output=output)


if __name__ == "__main__":
    unittest.main()
