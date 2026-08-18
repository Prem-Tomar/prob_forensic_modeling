import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralExperimentTests(unittest.TestCase):
    def test_cli_passes_explicit_paths_and_training_controls(self) -> None:
        from forensic_model.neural_experiment import main

        with patch("forensic_model.neural_experiment.run_real_image_experiment") as run:
            status = main(
                [
                    "--cifake-root",
                    "/data/cifake",
                    "--synthscars-root",
                    "/data/synthscars",
                    "--output",
                    "/reports/result.json",
                    "--checkpoint",
                    "/artifacts/model.pt",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "16",
                    "--seed",
                    "9",
                ]
            )

        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(positional, (Path("/data/cifake"), Path("/data/synthscars")))
        self.assertEqual(keywords["output"], Path("/reports/result.json"))
        self.assertEqual(keywords["checkpoint"], Path("/artifacts/model.pt"))
        self.assertEqual(keywords["training_config"].epochs, 3)
        self.assertEqual(keywords["training_config"].batch_size, 16)
        self.assertEqual(keywords["training_config"].seed, 9)


if __name__ == "__main__":
    unittest.main()
