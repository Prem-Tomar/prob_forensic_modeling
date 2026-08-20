import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class ImageStressExperimentTests(unittest.TestCase):
    def test_cli_passes_frozen_artifacts_and_evaluation_controls(self) -> None:
        from forensic_model.image_stress_experiment import main

        with patch("forensic_model.image_stress_experiment.run_image_stress_experiment") as run:
            status = main(
                [
                    "--cifake-root",
                    "/data/cifake",
                    "--neural-checkpoint",
                    "/artifacts/neural.pt",
                    "--feature-artifact",
                    "/artifacts/feature.json",
                    "--output",
                    "/reports/stress.json",
                    "--identity-policy",
                    "/data/reviewed-identities.json",
                    "--batch-size",
                    "64",
                    "--bootstrap-resamples",
                    "40",
                    "--cpu-threads",
                    "2",
                    "--seed",
                    "17",
                ]
            )

        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(positional, (Path("/data/cifake"),))
        self.assertEqual(keywords["neural_checkpoint"], Path("/artifacts/neural.pt"))
        self.assertEqual(keywords["feature_artifact"], Path("/artifacts/feature.json"))
        self.assertEqual(keywords["identity_policy"], Path("/data/reviewed-identities.json"))
        self.assertEqual(keywords["bootstrap_resamples"], 40)
        self.assertEqual(keywords["cpu_threads"], 2)

    def test_paired_shift_preserves_direction_and_magnitude(self) -> None:
        from forensic_model.image_stress_experiment import _paired_shift

        shift = _paired_shift([0.1, 0.5, 0.9], [0.2, 0.3, 0.9])

        self.assertAlmostEqual(shift["mean_signed"], -1.0 / 30.0)
        self.assertAlmostEqual(shift["mean_absolute"], 0.1)
        self.assertAlmostEqual(shift["maximum_absolute"], 0.2)

    def test_paired_shift_rejects_unaligned_vectors(self) -> None:
        from forensic_model.image_stress_experiment import _paired_shift

        with self.assertRaisesRegex(ValueError, "aligned"):
            _paired_shift([0.1], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
