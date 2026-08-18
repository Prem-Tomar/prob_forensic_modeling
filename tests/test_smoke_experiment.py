import json
import tempfile
import unittest
from pathlib import Path

from forensic_model.cli import main
from forensic_model.experiment import run_smoke_evaluation


class SmokeExperimentTests(unittest.TestCase):
    def test_report_is_deterministic_and_covers_every_phase(self) -> None:
        first = run_smoke_evaluation()
        second = run_smoke_evaluation()
        self.assertEqual(first, second)
        self.assertEqual(first["claim_scope"], "pipeline_mechanics_only")
        self.assertEqual(set(first["image"]["unseen_generators"]), {"unseen-blocks", "unseen-stripes"})
        self.assertIn("noise_0.02", first["image"]["postprocessing"])
        self.assertIn("frame_aggregation_baseline", first["video"])
        self.assertGreaterEqual(first["video"]["temporal_auroc_gain"], 0.0)

    def test_cli_writes_the_same_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            self.assertEqual(main(["smoke-evaluate", "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), run_smoke_evaluation())


if __name__ == "__main__":
    unittest.main()
