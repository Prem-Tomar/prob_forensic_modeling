import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class FeatureBaselineExperimentTests(unittest.TestCase):
    def test_cli_passes_paths_and_training_controls(self) -> None:
        from forensic_model.feature_baseline_experiment import main

        with patch("forensic_model.feature_baseline_experiment.run_feature_baseline_experiment") as run:
            status = main(
                [
                    "--cifake-root",
                    "/data/cifake",
                    "--synthscars-root",
                    "/data/synthscars",
                    "--output",
                    "/reports/feature.json",
                    "--artifact",
                    "/artifacts/feature.json",
                    "--neural-report",
                    "/reports/neural.json",
                    "--identity-policy",
                    "/data/reviewed-identities.json",
                    "--epochs",
                    "20",
                    "--batch-size",
                    "16",
                    "--cpu-threads",
                    "2",
                    "--seed",
                    "13",
                ]
            )

        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(positional, (Path("/data/cifake"), Path("/data/synthscars")))
        self.assertEqual(keywords["artifact"], Path("/artifacts/feature.json"))
        self.assertEqual(keywords["neural_report"], Path("/reports/neural.json"))
        self.assertEqual(keywords["identity_policy"], Path("/data/reviewed-identities.json"))
        self.assertEqual(keywords["training_config"].epochs, 20)
        self.assertEqual(keywords["training_config"].batch_size, 16)
        self.assertEqual(keywords["training_config"].cpu_threads, 2)
        self.assertEqual(keywords["training_config"].seed, 13)


if __name__ == "__main__":
    unittest.main()
