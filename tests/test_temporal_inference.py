import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "video extra is not installed")
class CalibratedTemporalInferenceTests(unittest.TestCase):
    def _checkpoint(self, path: Path, *, calibrated: bool = True) -> None:
        from forensic_model.neural_video import (
            TemporalConfig,
            create_scratch_temporal_detector,
            save_temporal_checkpoint,
        )

        model = create_scratch_temporal_detector(
            TemporalConfig(frame_widths=(4,), temporal_hidden=5, dropout=0.0),
            seed=23,
        )
        metadata = {}
        if calibrated:
            metadata = {
                "calibration_slope": 1.2,
                "calibration_intercept": -0.2,
                "threshold": 0.55,
            }
        save_temporal_checkpoint(path, model, metadata=metadata)

    def test_loads_checkpoint_and_returns_decomposed_calibrated_results(self) -> None:
        from forensic_model.temporal_inference import CalibratedTemporalDetector

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "temporal.pt"
            self._checkpoint(checkpoint)
            detector = CalibratedTemporalDetector.load(checkpoint, abstain_margin=0.0)
            results = detector.predict_tensors(torch.rand(2, 4, 3, 16, 16))

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0].frame_logits), 4)
        self.assertAlmostEqual(
            results[0].raw_score,
            results[0].frame_contribution + results[0].temporal_contribution,
            places=5,
        )
        self.assertIn(results[0].decision, {"camera_or_human", "synthetic"})
        self.assertFalse(results[0].abstained)

    def test_rejects_uncalibrated_checkpoint_and_invalid_clip_shape(self) -> None:
        from forensic_model.temporal_inference import CalibratedTemporalDetector

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "temporal.pt"
            self._checkpoint(checkpoint, calibrated=False)
            with self.assertRaisesRegex(ValueError, "calibrated inference"):
                CalibratedTemporalDetector.load(checkpoint)

            self._checkpoint(checkpoint)
            detector = CalibratedTemporalDetector.load(checkpoint)
            with self.assertRaisesRegex(ValueError, "non-empty batch"):
                detector.predict_tensors(torch.empty(0, 4, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
