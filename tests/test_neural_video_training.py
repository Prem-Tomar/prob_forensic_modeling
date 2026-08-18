import unittest
from types import SimpleNamespace

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object


class PatternVideoDataset(Dataset):
    def __init__(self, count: int) -> None:
        self.rows = []
        self.examples = []
        for index in range(count):
            label = index % 2
            frames = torch.full((5, 3, 16, 16), 0.4 + 0.01 * (index % 3))
            if label:
                frames[1::2, :, ::2, ::2] = 1.0
                frames[1::2, :, 1::2, 1::2] = 0.0
            self.rows.append((frames, label, f"group-{index}"))
            self.examples.append(SimpleNamespace(label=label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class TemporalTrainingTests(unittest.TestCase):
    def test_tiny_temporal_model_learns_from_scratch(self) -> None:
        from forensic_model.neural_video import TemporalConfig
        from forensic_model.neural_video_training import (
            TemporalTrainingConfig,
            evaluate_temporal_detector,
            train_temporal_detector,
        )

        detector = train_temporal_detector(
            PatternVideoDataset(40),
            PatternVideoDataset(20),
            training_config=TemporalTrainingConfig(
                seed=13,
                epochs=12,
                batch_size=8,
                learning_rate=0.01,
                patience=4,
                cpu_threads=1,
            ),
            model_config=TemporalConfig(frame_widths=(4, 8), temporal_hidden=6, dropout=0.0),
        )
        metrics = evaluate_temporal_detector(detector, PatternVideoDataset(20))

        self.assertGreater(metrics.auroc, 0.95)
        self.assertEqual(detector.metadata()["initialization"], "scratch")
        self.assertLessEqual(len(detector.history), 12)
        self.assertGreater(detector.threshold, 0.0)
        self.assertLess(detector.threshold, 1.0)

    def test_training_requires_both_labels(self) -> None:
        from forensic_model.neural_video import TemporalConfig
        from forensic_model.neural_video_training import TemporalTrainingConfig, train_temporal_detector

        dataset = PatternVideoDataset(1)
        with self.assertRaisesRegex(ValueError, "both labels"):
            train_temporal_detector(
                dataset,
                dataset,
                training_config=TemporalTrainingConfig(epochs=1, patience=1, cpu_threads=1),
                model_config=TemporalConfig(frame_widths=(4,), temporal_hidden=4, dropout=0.0),
            )


if __name__ == "__main__":
    unittest.main()
