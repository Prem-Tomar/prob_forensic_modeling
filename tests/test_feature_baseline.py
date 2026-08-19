import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class FeatureBaselineTests(unittest.TestCase):
    def test_vectorized_features_match_dependency_free_reference(self) -> None:
        from forensic_model.feature_baseline import extract_summary_feature_matrix
        from forensic_model.features import extract_features
        from forensic_model.image import RGBImage

        image = torch.rand(3, 8, 9)
        rows = image.permute(1, 2, 0).tolist()
        expected = extract_features(RGBImage.from_rows(rows)).values
        actual = extract_summary_feature_matrix(image.unsqueeze(0))[0].tolist()

        for first, second in zip(expected, actual):
            self.assertAlmostEqual(first, second, places=10)

    def test_vectorized_optimizer_learns_and_returns_public_model(self) -> None:
        from forensic_model.feature_baseline import FeatureTrainingConfig, fit_feature_detector
        from forensic_model.features import FeatureVector

        negative = torch.zeros(12, 10, dtype=torch.float64)
        positive = torch.ones(12, 10, dtype=torch.float64)
        training = torch.cat((negative, positive))
        labels = [0] * 12 + [1] * 12
        result = fit_feature_detector(
            training,
            labels,
            training,
            labels,
            config=FeatureTrainingConfig(epochs=100, learning_rate=0.2, batch_size=8, cpu_threads=1),
        )

        low = result.detector.model.predict(
            FeatureVector(result.detector.model.feature_names, tuple(0.0 for _ in range(10)))
        )
        high = result.detector.model.predict(
            FeatureVector(result.detector.model.feature_names, tuple(1.0 for _ in range(10)))
        )
        self.assertGreater(high.probability_synthetic, low.probability_synthetic)
        self.assertLess(result.final_training_loss, 0.2)


if __name__ == "__main__":
    unittest.main()
