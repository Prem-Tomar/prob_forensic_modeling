import unittest
from pathlib import Path
from types import SimpleNamespace
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
                    "--identity-policy",
                    "/data/reviewed-identities.json",
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
        self.assertEqual(keywords["identity_policy"], Path("/data/reviewed-identities.json"))
        self.assertEqual(keywords["training_config"].epochs, 3)
        self.assertEqual(keywords["training_config"].batch_size, 16)
        self.assertEqual(keywords["training_config"].seed, 9)

    def test_branch_summary_distinguishes_raw_and_deployed_frequency_evidence(self) -> None:
        from forensic_model.neural_experiment import _branch_summary

        class Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.zeros(3, 8, 8)
                image[0, 0, 0] = index + 1
                image[0, 0, 1] = (index + 1) * 4
                return image, 0, str(index)

        class Model:
            def eval(self):
                return self

            def forward_evidence(self, images):
                return SimpleNamespace(
                    spatial_contribution=images[:, 0, 0, 0],
                    frequency_contribution=images[:, 0, 0, 1],
                )

        summary = _branch_summary(Model(), Dataset(), frequency_weight=0.25)

        self.assertEqual(summary["selected_frequency_weight"], 0.25)
        self.assertEqual(summary["mean_frequency_contribution"], 6.0)
        self.assertEqual(summary["mean_deployed_frequency_contribution"], 1.5)
        self.assertEqual(summary["mean_absolute_deployed_frequency_contribution"], 1.5)


if __name__ == "__main__":
    unittest.main()
