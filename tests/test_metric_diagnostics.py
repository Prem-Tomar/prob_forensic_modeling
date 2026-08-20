import unittest

from forensic_model.metrics import classification_diagnostics, reliability_bins, wilson_interval


class MetricDiagnosticTests(unittest.TestCase):
    def test_reports_reliability_rates_and_selective_risk(self) -> None:
        diagnostics = classification_diagnostics(
            [0, 0, 1, 1],
            [0.1, 0.48, 0.52, 0.9],
            threshold=0.5,
            abstain_margin=0.05,
            calibration_bins=2,
        )

        self.assertEqual(len(diagnostics.reliability_bins), 2)
        self.assertEqual(diagnostics.sensitivity_wilson_95_percent.estimate, 1.0)
        self.assertEqual(diagnostics.specificity_wilson_95_percent.estimate, 1.0)
        self.assertEqual(diagnostics.selective.covered, 2)
        self.assertEqual(diagnostics.selective.coverage, 0.5)
        self.assertEqual(diagnostics.selective.selective_risk, 0.0)

    def test_handles_complete_abstention_without_inventing_risk(self) -> None:
        diagnostics = classification_diagnostics(
            [0, 1],
            [0.49, 0.51],
            threshold=0.5,
            abstain_margin=0.02,
        )
        self.assertEqual(diagnostics.selective.coverage, 0.0)
        self.assertIsNone(diagnostics.selective.selective_accuracy)
        self.assertIsNone(diagnostics.selective.selective_risk)

    def test_equal_mass_bins_are_deterministic_and_complete(self) -> None:
        bins = reliability_bins([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], bins=3)
        self.assertEqual(sum(item.count for item in bins), 4)
        self.assertEqual(bins[0].minimum_probability, 0.1)
        self.assertEqual(bins[-1].maximum_probability, 0.9)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        interval = wilson_interval(8, 10)
        self.assertLess(interval.lower, 0.8)
        self.assertGreater(interval.upper, 0.8)
        with self.assertRaisesRegex(ValueError, "Wilson"):
            wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
