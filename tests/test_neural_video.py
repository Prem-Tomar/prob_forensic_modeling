import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class TemporalResidualDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        from forensic_model.neural_video import TemporalConfig, create_scratch_temporal_detector

        self.config = TemporalConfig(frame_widths=(4, 8), temporal_hidden=6, dropout=0.0)
        self.model = create_scratch_temporal_detector(self.config, seed=23)

    def test_logit_decomposes_into_frame_and_temporal_evidence(self) -> None:
        frames = torch.rand(3, 5, 3, 16, 16)

        evidence = self.model.forward_evidence(frames)

        torch.testing.assert_close(evidence.logits, evidence.frame_contribution + evidence.temporal_contribution)
        self.assertEqual(tuple(evidence.logits.shape), (3,))
        self.assertEqual(tuple(evidence.frame_logits.shape), (3, 5))

    def test_scratch_initialization_is_repeatable(self) -> None:
        from forensic_model.neural_video import create_scratch_temporal_detector

        repeated = create_scratch_temporal_detector(self.config, seed=23)
        for first, second in zip(self.model.parameters(), repeated.parameters()):
            torch.testing.assert_close(first, second)

    def test_checkpoint_round_trip_uses_weights_only_format(self) -> None:
        from forensic_model.neural_video import load_temporal_checkpoint, save_temporal_checkpoint

        self.model.eval()
        frames = torch.rand(2, 4, 3, 16, 16)
        expected = self.model(frames)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporal.pt"
            save_temporal_checkpoint(path, self.model, metadata={"initialization": "scratch", "seed": 23})
            restored, metadata = load_temporal_checkpoint(path)
        restored.eval()

        torch.testing.assert_close(expected, restored(frames))
        self.assertEqual(metadata["initialization"], "scratch")

    def test_rejects_malformed_frame_sequences(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.rand(2, 2, 3, 16, 16))
        with self.assertRaises(ValueError):
            self.model(torch.rand(2, 4, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
