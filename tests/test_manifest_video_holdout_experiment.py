import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "video extra is not installed")
class ManifestVideoHoldoutTests(unittest.TestCase):
    def test_scores_temporal_probabilities_and_explanation_contributions(self) -> None:
        from forensic_model.manifest_video_holdout_experiment import _score_temporal

        class Detector:
            def predict_tensors(self, clips):
                return tuple(
                    SimpleNamespace(
                        probability_synthetic=float(clip.mean()),
                        frame_contribution=float(clip[:, 0].mean()),
                        temporal_contribution=float(clip[:, 1].mean()),
                    )
                    for clip in clips
                )

        rows = [
            (torch.zeros(4, 3, 8, 8), 0, "content-a"),
            (torch.ones(4, 3, 8, 8), 1, "content-a"),
        ]

        probabilities, labels, groups, explanation = _score_temporal(Detector(), rows, batch_size=2)

        self.assertEqual(probabilities, [0.0, 1.0])
        self.assertEqual(labels, [0, 1])
        self.assertEqual(groups, ["content-a", "content-a"])
        self.assertEqual(explanation["sample_count"], 2)
        self.assertEqual(explanation["mean_absolute_frame_contribution"], 0.5)
        self.assertEqual(explanation["mean_absolute_temporal_contribution"], 0.5)


if __name__ == "__main__":
    unittest.main()
