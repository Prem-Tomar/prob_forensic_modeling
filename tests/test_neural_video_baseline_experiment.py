import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "video extra is not installed")
class FrameAggregationExperimentTests(unittest.TestCase):
    def test_cli_passes_paths_and_evaluation_controls(self) -> None:
        from forensic_model.neural_video_baseline_experiment import main

        with patch(
            "forensic_model.neural_video_baseline_experiment.run_frame_aggregation_experiment"
        ) as run:
            status = main(
                [
                    "--davis-root",
                    "/data/davis",
                    "--keling-root",
                    "/data/keling",
                    "--sora-root",
                    "/data/sora",
                    "--image-checkpoint",
                    "/artifacts/image.pt",
                    "--output",
                    "/reports/frames.json",
                    "--frame-count",
                    "6",
                    "--image-size",
                    "24",
                    "--batch-size",
                    "4",
                    "--seed",
                    "31",
                ]
            )

        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(
            positional,
            (
                Path("/data/davis"),
                Path("/data/keling"),
                Path("/data/sora"),
                Path("/artifacts/image.pt"),
            ),
        )
        self.assertEqual(keywords["output"], Path("/reports/frames.json"))
        self.assertEqual(keywords["frame_count"], 6)
        self.assertEqual(keywords["image_size"], 24)
        self.assertEqual(keywords["batch_size"], 4)
        self.assertEqual(keywords["seed"], 31)


if __name__ == "__main__":
    unittest.main()
