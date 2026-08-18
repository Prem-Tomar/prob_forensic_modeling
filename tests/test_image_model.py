import tempfile
import unittest
import math
from pathlib import Path

from forensic_model.detector import ImageDetector
from forensic_model.features import FEATURE_NAMES, extract_features
from forensic_model.image import PPMDecoder, RGBImage


def smooth(level: float) -> RGBImage:
    return RGBImage.from_rows([[(level, level, level) for _ in range(4)] for _ in range(4)])


def checker(low: float, high: float) -> RGBImage:
    return RGBImage.from_rows(
        [[((high if (x + y) % 2 else low),) * 3 for x in range(4)] for y in range(4)]
    )


class FeatureTests(unittest.TestCase):
    def test_checkerboard_has_more_frequency_energy_than_smooth_image(self) -> None:
        smooth_features = extract_features(smooth(0.4)).as_dict()
        checker_features = extract_features(checker(0.2, 0.8)).as_dict()
        self.assertGreater(checker_features["checkerboard_energy"], smooth_features["checkerboard_energy"])
        self.assertGreater(checker_features["laplacian_energy"], smooth_features["laplacian_energy"])
        self.assertEqual(tuple(smooth_features), FEATURE_NAMES)

    def test_ppm_decoder_reads_binary_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ppm"
            path.write_bytes(b"P6\n2 2\n255\n" + bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]))
            image = PPMDecoder().decode(path)
            self.assertEqual((image.width, image.height), (2, 2))
            self.assertEqual(image.pixel(1, 0), (1.0, 0.0, 0.0))

    def test_ppm_decoder_preserves_whitespace_valued_first_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ppm"
            path.write_bytes(b"P6\n2 2\n255\n" + bytes([10, 20, 30] * 4))
            image = PPMDecoder().decode(path)
            self.assertAlmostEqual(image.pixel(0, 0)[0], 10 / 255)

    def test_rgb_image_rejects_non_finite_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            RGBImage(2, 2, ((math.nan, 0.0, 0.0),) * 4)


class DetectorTests(unittest.TestCase):
    def test_model_learns_and_round_trips(self) -> None:
        images = [smooth(0.2), smooth(0.4), smooth(0.6), checker(0.1, 0.9), checker(0.2, 0.8), checker(0.3, 0.7)]
        detector = ImageDetector.train(images, [0, 0, 0, 1, 1, 1])
        real_prediction = detector.predict_image(smooth(0.5))
        synthetic_prediction = detector.predict_image(checker(0.25, 0.75))
        self.assertLess(real_prediction.probability_synthetic, 0.5)
        self.assertGreater(synthetic_prediction.probability_synthetic, 0.5)
        self.assertEqual(set(synthetic_prediction.contributions), set(FEATURE_NAMES))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            detector.save(path)
            restored = ImageDetector.load(path)
            self.assertAlmostEqual(
                restored.predict_image(checker(0.25, 0.75)).probability_synthetic,
                synthetic_prediction.probability_synthetic,
            )

    def test_model_loader_rejects_invalid_scales(self) -> None:
        detector = ImageDetector.train([smooth(0.2), checker(0.1, 0.9)], [0, 1])
        values = detector.model.to_dict()
        values["scales"] = [0.0] * len(FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, "scales must be positive"):
            detector.model.from_dict(values)


if __name__ == "__main__":
    unittest.main()
