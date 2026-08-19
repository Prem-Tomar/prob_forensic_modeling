import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class CalibratedNeuralImageDetectorTests(unittest.TestCase):
    def _checkpoint(self, path: Path, *, frequency_weight: float = 0.25) -> None:
        from forensic_model.neural import NeuralConfig, create_scratch_detector, save_neural_checkpoint

        model = create_scratch_detector(
            NeuralConfig(spatial_widths=(4,), frequency_widths=(4,), dropout=0.0),
            seed=37,
        )
        save_neural_checkpoint(
            path,
            model,
            metadata={
                "calibration_slope": 1.1,
                "calibration_intercept": -0.2,
                "threshold": 0.55,
                "frequency_weight": frequency_weight,
            },
        )

    def test_loads_checkpoint_and_returns_explainable_calibrated_results(self) -> None:
        from forensic_model.neural_inference import CalibratedNeuralImageDetector

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "image.pt"
            self._checkpoint(checkpoint)
            detector = CalibratedNeuralImageDetector.load(checkpoint)
            results = detector.predict_tensors(torch.rand(3, 3, 16, 16))

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertAlmostEqual(
                result.raw_score,
                result.spatial_contribution + result.deployed_frequency_contribution,
            )
            self.assertAlmostEqual(
                result.deployed_frequency_contribution,
                0.25 * result.raw_frequency_contribution,
            )
            self.assertIn(result.decision, {"camera_or_human", "synthetic", "abstain"})
            self.assertGreaterEqual(result.probability_synthetic, 0.0)
            self.assertLessEqual(result.probability_synthetic, 1.0)

    def test_predict_file_uses_the_same_tensor_contract(self) -> None:
        from PIL import Image

        from forensic_model.neural_inference import CalibratedNeuralImageDetector

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "image.pt"
            image = root / "sample.png"
            self._checkpoint(checkpoint, frequency_weight=0.0)
            Image.new("RGB", (20, 18), (40, 100, 180)).save(image)

            result = CalibratedNeuralImageDetector.load(checkpoint).predict_file(image, image_size=16)

        self.assertEqual(result.deployed_frequency_contribution, 0.0)

    def test_rejects_missing_metadata_and_invalid_batches(self) -> None:
        from forensic_model.neural import create_scratch_detector, save_neural_checkpoint
        from forensic_model.neural_inference import CalibratedNeuralImageDetector

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "image.pt"
            save_neural_checkpoint(checkpoint, create_scratch_detector(seed=5), metadata={})
            with self.assertRaisesRegex(ValueError, "calibrated inference"):
                CalibratedNeuralImageDetector.load(checkpoint)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "image.pt"
            self._checkpoint(checkpoint)
            detector = CalibratedNeuralImageDetector.load(checkpoint)
            with self.assertRaisesRegex(ValueError, "non-empty batch"):
                detector.predict_tensors(torch.empty(0, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
