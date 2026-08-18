import tempfile
import unittest
from pathlib import Path

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy, decide
from forensic_model.detector import ImageDetector
from forensic_model.image import RGBImage
from forensic_model.model import Prediction


def image(level: float, alternating: bool) -> RGBImage:
    rows = []
    for y in range(4):
        row = []
        for x in range(4):
            value = (1.0 - level if (x + y) % 2 else level) if alternating else level
            row.append((value, value, value))
        rows.append(row)
    return RGBImage.from_rows(rows)


class ConfidenceTests(unittest.TestCase):
    def test_calibrator_orders_scores_and_round_trips(self) -> None:
        calibrator = PlattCalibrator.fit([-3.0, -1.0, 1.0, 3.0], [0, 0, 1, 1])
        self.assertLess(calibrator.transform(-1.0), calibrator.transform(1.0))
        self.assertEqual(PlattCalibrator.from_dict(calibrator.to_dict()), calibrator)

    def test_policy_abstains_near_threshold_and_explains(self) -> None:
        prediction = Prediction(0.53, 0.12, {"noise": 0.4, "color": -0.1})
        result = decide(prediction, calibrator=None, policy=DecisionPolicy(abstain_margin=0.05))
        self.assertEqual(result.decision, "abstain")
        self.assertTrue(result.abstained)
        self.assertEqual(result.reasons[0].feature, "noise")

    def test_detector_calibration_survives_serialization(self) -> None:
        training = [image(0.2, False), image(0.4, False), image(0.1, True), image(0.3, True)]
        detector = ImageDetector.train(training, [0, 0, 1, 1])
        calibration = [image(0.25, False), image(0.45, False), image(0.15, True), image(0.35, True)]
        calibrated = detector.fit_calibrator(calibration, [0, 0, 1, 1])
        before = calibrated.analyze_image(image(0.2, True))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.json"
            calibrated.save(path)
            restored = ImageDetector.load(path)
            after = restored.analyze_image(image(0.2, True))

        self.assertEqual(after.calibration, "platt")
        self.assertAlmostEqual(after.probability_synthetic, before.probability_synthetic)


if __name__ == "__main__":
    unittest.main()
