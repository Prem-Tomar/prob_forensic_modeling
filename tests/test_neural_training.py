import unittest

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object


class PatternDataset(Dataset):
    def __init__(self, count: int) -> None:
        self.rows = []
        for index in range(count):
            label = index % 2
            image = torch.zeros(3, 16, 16)
            if label:
                image[:, ::2, ::2] = 1.0
                image[:, 1::2, 1::2] = 1.0
            else:
                image.fill_(0.45 + (index % 4) * 0.01)
            self.rows.append((image, label, f"group-{index}"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralTrainingTests(unittest.TestCase):
    def test_threshold_handles_ties(self) -> None:
        from forensic_model.neural_training import balanced_accuracy_threshold

        self.assertEqual(balanced_accuracy_threshold([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]), 0.9)

    def test_tiny_detector_learns_without_pretrained_weights(self) -> None:
        from forensic_model.neural import NeuralConfig
        from forensic_model.neural_training import TrainingConfig, evaluate_neural_detector, train_neural_detector

        detector = train_neural_detector(
            PatternDataset(32),
            PatternDataset(16),
            training_config=TrainingConfig(seed=7, epochs=4, batch_size=8, learning_rate=0.01, cpu_threads=1),
            model_config=NeuralConfig(spatial_widths=(4, 8), frequency_widths=(4, 8), dropout=0.0),
        )
        metrics = evaluate_neural_detector(detector, PatternDataset(16))
        self.assertGreater(metrics.auroc, 0.95)
        self.assertEqual(detector.metadata()["initialization"], "scratch")

    def test_recalibrates_after_validation_selected_blend(self) -> None:
        from forensic_model.neural import NeuralConfig
        from forensic_model.neural_training import (
            TrainingConfig,
            recalibrate_neural_detector,
            train_neural_detector,
        )

        validation = PatternDataset(16)
        detector = train_neural_detector(
            PatternDataset(32),
            validation,
            training_config=TrainingConfig(seed=11, epochs=2, batch_size=8, learning_rate=0.01, cpu_threads=1),
            model_config=NeuralConfig(spatial_widths=(4, 8), frequency_widths=(4, 8), dropout=0.0),
        )
        recalibrated = recalibrate_neural_detector(detector, validation, frequency_weight=0.25)

        self.assertEqual(recalibrated.frequency_weight, 0.25)
        self.assertEqual(recalibrated.metadata()["frequency_weight"], 0.25)
        self.assertGreater(recalibrated.threshold, 0.0)
        self.assertLess(recalibrated.threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
