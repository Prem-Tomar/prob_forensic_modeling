import unittest

from forensic_model.slice_evaluation import evaluate_slices


class SliceEvaluationTests(unittest.TestCase):
    def test_reports_evaluated_and_insufficient_group_slices(self) -> None:
        labels = [0, 1, 0, 1, 0, 1]
        probabilities = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        groups = ["a", "a", "b", "b", "c", "c"]
        values = ["broad", "broad", "broad", "broad", "narrow", "narrow"]

        report = evaluate_slices(
            labels,
            probabilities,
            groups,
            values,
            threshold=0.5,
            bootstrap_resamples=20,
            seed=7,
        )

        self.assertEqual(report["broad"]["status"], "evaluated")
        self.assertEqual(report["broad"]["metrics"]["auroc"], 1.0)
        self.assertEqual(report["narrow"]["status"], "insufficient_group_coverage")
        self.assertIsNone(report["narrow"]["auroc_grouped_bootstrap_95_percent"])

    def test_rejects_misalignment_and_marks_single_label_slice(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            evaluate_slices([0], [0.1], ["a"], [], threshold=0.5, bootstrap_resamples=20, seed=1)

        report = evaluate_slices(
            [0, 1],
            [0.1, 0.9],
            ["a", "a"],
            ["real-only", ""],
            threshold=0.5,
            bootstrap_resamples=20,
            seed=1,
        )
        self.assertEqual(report["real-only"]["status"], "insufficient_label_coverage")

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration"):
            evaluate_slices([], [], [], [], threshold=1.0, bootstrap_resamples=20, seed=1)
        with self.assertRaisesRegex(ValueError, "configuration"):
            evaluate_slices([], [], [], [], threshold=0.5, bootstrap_resamples=19, seed=1)


if __name__ == "__main__":
    unittest.main()
