import tempfile
import unittest
import hashlib
import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class FrozenFrameAggregatorTests(unittest.TestCase):
    def _checkpoint(self, path: Path):
        from forensic_model.neural import NeuralConfig, create_scratch_detector, save_neural_checkpoint

        model = create_scratch_detector(
            NeuralConfig(spatial_widths=(4,), frequency_widths=(4,), dropout=0.0),
            seed=29,
        )
        save_neural_checkpoint(
            path,
            model,
            metadata={
                "calibration_slope": 1.2,
                "calibration_intercept": -0.1,
                "threshold": 0.6,
                "frequency_weight": 0.25,
            },
        )

    def test_loads_calibrated_checkpoint_and_averages_frames(self) -> None:
        from forensic_model.neural_video_baseline import FrozenFrameAggregator

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            self._checkpoint(path)
            aggregator = FrozenFrameAggregator.load(path)

        result = aggregator.predict(torch.rand(5, 3, 16, 16))

        self.assertEqual(len(result.frame_probabilities), 5)
        self.assertAlmostEqual(
            result.probability_synthetic,
            sum(result.frame_probabilities) / 5,
        )
        self.assertEqual(aggregator.threshold, 0.6)
        self.assertEqual(aggregator.frequency_weight, 0.25)

    def test_fits_clip_calibration_on_validation_only(self) -> None:
        from forensic_model.neural_video_baseline import (
            FrozenFrameAggregator,
            fit_clip_aggregation,
            score_validated_frame_aggregation,
        )

        rows = []
        for index in range(8):
            label = index % 2
            rows.append((torch.full((4, 3, 16, 16), 0.2 + 0.6 * label), label, f"group-{index}"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            self._checkpoint(path)
            calibrated = fit_clip_aggregation(FrozenFrameAggregator.load(path), rows, batch_size=4)
            probabilities, labels, groups = score_validated_frame_aggregation(calibrated, rows, batch_size=4)

        self.assertEqual(len(probabilities), 8)
        self.assertEqual(labels, [0, 1] * 4)
        self.assertEqual(len(groups), 8)
        self.assertGreater(calibrated.threshold, 0.0)
        self.assertLess(calibrated.threshold, 1.0)

    def test_rejects_checkpoint_without_calibration(self) -> None:
        from forensic_model.neural import create_scratch_detector, save_neural_checkpoint
        from forensic_model.neural_video_baseline import FrozenFrameAggregator

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            save_neural_checkpoint(path, create_scratch_detector(seed=3), metadata={})
            with self.assertRaisesRegex(ValueError, "calibrated inference"):
                FrozenFrameAggregator.load(path)

    def test_restores_clip_calibration_only_for_bound_checkpoint(self) -> None:
        from forensic_model.checkpoint_digest import semantic_checkpoint_sha256
        from forensic_model.neural_video_baseline import ValidatedFrameAggregator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "image.pt"
            report = root / "frames.json"
            self._checkpoint(checkpoint)
            report.write_text(
                json.dumps(
                    {
                        "image_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "image_checkpoint_semantic_sha256": semantic_checkpoint_sha256(checkpoint),
                        "clip_calibration": {"method": "platt", "slope": 1.4, "intercept": -0.3},
                        "threshold_selected_on_validation": 0.62,
                    }
                ),
                encoding="utf-8",
            )

            restored = ValidatedFrameAggregator.load(checkpoint, report)

        self.assertEqual(restored.threshold, 0.62)
        self.assertEqual(restored.clip_calibrator.slope, 1.4)

    def test_rejects_frame_calibration_bound_to_another_checkpoint(self) -> None:
        from forensic_model.neural_video_baseline import ValidatedFrameAggregator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "image.pt"
            report = root / "frames.json"
            self._checkpoint(checkpoint)
            report.write_text(
                json.dumps(
                    {
                        "image_checkpoint_sha256": "0" * 64,
                        "image_checkpoint_semantic_sha256": "0" * 64,
                        "clip_calibration": {"method": "platt", "slope": 1.0, "intercept": 0.0},
                        "threshold_selected_on_validation": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different checkpoint file"):
                ValidatedFrameAggregator.load(checkpoint, report)


if __name__ == "__main__":
    unittest.main()
