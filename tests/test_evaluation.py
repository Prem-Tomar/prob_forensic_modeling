import unittest

from forensic_model.detector import ImageDetector
from forensic_model.image import RGBImage
from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.robustness import (
    EvaluationSample,
    add_noise,
    evaluate_postprocessing,
    evaluate_unseen_generators,
)


def smooth(level: float) -> RGBImage:
    return RGBImage.from_rows([[(level, level, level) for _ in range(6)] for _ in range(6)])


def pattern(level: float, period: int) -> RGBImage:
    return RGBImage.from_rows(
        [[((1.0 - level if (x + y) % period == 0 else level),) * 3 for x in range(6)] for y in range(6)]
    )


class MetricTests(unittest.TestCase):
    def test_perfect_ranking_metrics(self) -> None:
        metrics = binary_metrics([0, 0, 1, 1], [0.05, 0.2, 0.8, 0.95])
        self.assertEqual(metrics.auroc, 1.0)
        self.assertEqual(metrics.auprc, 1.0)
        self.assertEqual(metrics.balanced_accuracy, 1.0)
        self.assertEqual(metrics.false_positive_rate_at_90_recall, 0.0)
        self.assertEqual(metrics.recall_at_1_percent_false_positive_rate, 1.0)

    def test_auroc_handles_ties_without_row_order_bias(self) -> None:
        first = auroc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9])
        second = auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9])
        self.assertEqual(first, second)

    def test_grouped_bootstrap_is_reproducible(self) -> None:
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
        groups = [f"group-{index}" for index in range(6)]
        statistic = lambda y, p: auroc(y, p)
        first = grouped_bootstrap_interval(labels, scores, groups, statistic=statistic, resamples=100, seed=7)
        second = grouped_bootstrap_interval(labels, scores, groups, statistic=statistic, resamples=100, seed=7)
        self.assertEqual(first, second)


class RobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = ImageDetector.train(
            [smooth(0.2), smooth(0.4), pattern(0.2, 2), pattern(0.3, 2)],
            [0, 0, 1, 1],
        )
        self.samples = [
            EvaluationSample(smooth(0.3), 0, "real-a"),
            EvaluationSample(smooth(0.5), 0, "real-b"),
            EvaluationSample(pattern(0.25, 3), 1, "generated-a", "unseen-a"),
            EvaluationSample(pattern(0.35, 4), 1, "generated-b", "unseen-b"),
        ]

    def test_noise_transform_is_deterministic(self) -> None:
        self.assertEqual(add_noise(self.samples[0].image, 0.02, seed=4), add_noise(self.samples[0].image, 0.02, seed=4))

    def test_unseen_generator_slices_reject_training_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "seen during training"):
            evaluate_unseen_generators(self.detector, self.samples, trained_generator_families={"unseen-a"})
        results = evaluate_unseen_generators(self.detector, self.samples, trained_generator_families={"known"})
        self.assertEqual(set(results), {"unseen-a", "unseen-b"})

    def test_postprocessing_reports_clean_and_stressed_metrics(self) -> None:
        results = evaluate_postprocessing(self.detector, self.samples)
        self.assertIn("clean", results)
        self.assertIn("noise_0.02", results)
        self.assertTrue(all(metrics.count == 4 for metrics in results.values()))


if __name__ == "__main__":
    unittest.main()
