import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        from forensic_model.neural import NeuralConfig, create_scratch_detector

        self.config = NeuralConfig(spatial_widths=(4, 8), frequency_widths=(4, 6), dropout=0.0)
        self.model = create_scratch_detector(self.config, seed=17)

    def test_logit_is_sum_of_explainable_branch_contributions(self) -> None:
        images = torch.rand(3, 3, 16, 16)
        evidence = self.model.forward_evidence(images)
        torch.testing.assert_close(evidence.logits, evidence.spatial_contribution + evidence.frequency_contribution)
        self.assertEqual(tuple(evidence.logits.shape), (3,))

    def test_scratch_initialization_is_repeatable(self) -> None:
        from forensic_model.neural import create_scratch_detector

        repeated = create_scratch_detector(self.config, seed=17)
        for first, second in zip(self.model.parameters(), repeated.parameters()):
            torch.testing.assert_close(first, second)

    def test_checkpoint_round_trip_uses_versioned_weights(self) -> None:
        from forensic_model.neural import load_neural_checkpoint, save_neural_checkpoint

        self.model.eval()
        images = torch.rand(2, 3, 16, 16)
        expected = self.model(images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            save_neural_checkpoint(path, self.model, metadata={"initialization": "scratch", "seed": 17})
            restored, metadata = load_neural_checkpoint(path)
        restored.eval()
        torch.testing.assert_close(expected, restored(images))
        self.assertEqual(metadata["initialization"], "scratch")


if __name__ == "__main__":
    unittest.main()
