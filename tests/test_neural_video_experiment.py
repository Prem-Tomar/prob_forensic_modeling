import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "video extra is not installed")
class TemporalExperimentTests(unittest.TestCase):
    def test_cli_passes_paths_and_training_controls(self) -> None:
        from forensic_model.neural_video_experiment import main

        with patch("forensic_model.neural_video_experiment.run_real_video_experiment") as run:
            status = main(
                [
                    "--davis-root",
                    "/data/davis",
                    "--keling-root",
                    "/data/keling",
                    "--sora-root",
                    "/data/sora",
                    "--output",
                    "/reports/video.json",
                    "--checkpoint",
                    "/artifacts/video.pt",
                    "--epochs",
                    "7",
                    "--batch-size",
                    "4",
                    "--patience",
                    "3",
                    "--frame-count",
                    "6",
                    "--image-size",
                    "24",
                    "--seed",
                    "17",
                ]
            )

        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(
            positional,
            (Path("/data/davis"), Path("/data/keling"), Path("/data/sora")),
        )
        self.assertEqual(keywords["output"], Path("/reports/video.json"))
        self.assertEqual(keywords["checkpoint"], Path("/artifacts/video.pt"))
        self.assertEqual(keywords["training_config"].epochs, 7)
        self.assertEqual(keywords["training_config"].batch_size, 4)
        self.assertEqual(keywords["training_config"].patience, 3)
        self.assertEqual(keywords["frame_count"], 6)
        self.assertEqual(keywords["image_size"], 24)


if __name__ == "__main__":
    unittest.main()
