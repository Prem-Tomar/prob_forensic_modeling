import unittest
import tempfile
from pathlib import Path

from forensic_model.detector import ImageDetector
from forensic_model.image import RGBImage
from forensic_model.video import TimedFrame, VideoClip, VideoDetector, sample_frames


def smooth(level: float) -> RGBImage:
    return RGBImage.from_rows([[(level, level, level) for _ in range(6)] for _ in range(6)])


def patterned(level: float, period: int = 2) -> RGBImage:
    return RGBImage.from_rows(
        [[((1.0 - level if (x + y) % period == 0 else level),) * 3 for x in range(6)] for y in range(6)]
    )


def clip(clip_id: str, frames: list[RGBImage]) -> VideoClip:
    return VideoClip(clip_id, tuple(TimedFrame(float(index), image) for index, image in enumerate(frames)))


class VideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_detector = ImageDetector.train(
            [smooth(0.2), smooth(0.4), patterned(0.2), patterned(0.3)],
            [0, 0, 1, 1],
        )

    def test_sampling_keeps_endpoints_and_scene_change(self) -> None:
        source = clip("scene", [smooth(0.1)] * 4 + [smooth(0.9)] * 4)
        sampled = sample_frames(source, 3)
        self.assertEqual(sampled[0].timestamp_seconds, 0.0)
        self.assertEqual(sampled[-1].timestamp_seconds, 7.0)
        self.assertIn(4.0, [frame.timestamp_seconds for frame in sampled])

    def test_temporal_model_learns_and_explains_clip(self) -> None:
        real_clips = [
            clip("real-a", [smooth(0.2), smooth(0.25), smooth(0.3), smooth(0.35)]),
            clip("real-b", [smooth(0.4), smooth(0.42), smooth(0.44), smooth(0.46)]),
        ]
        synthetic_clips = [
            clip("synthetic-a", [patterned(0.2), smooth(0.3), patterned(0.25), smooth(0.35)]),
            clip("synthetic-b", [patterned(0.3, 3), smooth(0.4), patterned(0.2, 3), smooth(0.3)]),
        ]
        detector = VideoDetector.train(self.image_detector, real_clips + synthetic_clips, [0, 0, 1, 1])
        result = detector.analyze_clip(
            clip("candidate", [patterned(0.25), smooth(0.35), patterned(0.2), smooth(0.4)])
        )
        self.assertGreater(result.decision.probability_synthetic, 0.5)
        self.assertEqual(len(result.frame_scores), 4)
        self.assertTrue(result.notable_timestamps)
        self.assertGreaterEqual(result.aggregation_probability, 0.0)
        self.assertIn("motion_residual_mean", result.temporal_features.names)
        self.assertIn("luma_flicker_energy", result.temporal_features.names)

    def test_clip_calibration_is_explicit(self) -> None:
        real = [
            clip("real-a", [smooth(0.2), smooth(0.25), smooth(0.3)]),
            clip("real-b", [smooth(0.4), smooth(0.42), smooth(0.44)]),
        ]
        synthetic = [
            clip("synthetic-a", [patterned(0.2), smooth(0.3), patterned(0.25)]),
            clip("synthetic-b", [patterned(0.3), smooth(0.4), patterned(0.2)]),
        ]
        detector = VideoDetector.train(self.image_detector, real + synthetic, [0, 0, 1, 1])
        calibrated = detector.fit_calibrator(real + synthetic, [0, 0, 1, 1])
        self.assertEqual(calibrated.analyze_clip(synthetic[0]).decision.calibration, "platt")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video-detector.json"
            calibrated.save(path)
            restored = VideoDetector.load(path)
            self.assertAlmostEqual(
                restored.analyze_clip(synthetic[0]).decision.probability_synthetic,
                calibrated.analyze_clip(synthetic[0]).decision.probability_synthetic,
            )


if __name__ == "__main__":
    unittest.main()
