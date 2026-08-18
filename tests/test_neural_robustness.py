import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


class BranchDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        spatial, frequency, label = self.rows[index]
        image = torch.zeros(3, 8, 8)
        image[0, 0, 0] = spatial
        image[0, 0, 1] = frequency
        return image, label, f"sample-{index}"


class EncodedBranchModel:
    def eval(self):
        return self

    def forward_evidence(self, images):
        return SimpleNamespace(
            spatial_contribution=images[:, 0, 0, 0],
            frequency_contribution=images[:, 0, 0, 1],
        )


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class BranchRobustnessTests(unittest.TestCase):
    def setUp(self):
        from forensic_model.neural_robustness import select_frequency_weight

        self.select = select_frequency_weight

    def test_selects_weight_using_worst_validation_slice(self):
        clean = BranchDataset(((-2.0, -1.0, 0), (-1.0, -0.5, 0), (1.0, 0.5, 1), (2.0, 1.0, 1)))
        resized = BranchDataset(((-2.0, 4.0, 0), (-1.0, 3.0, 0), (1.0, -3.0, 1), (2.0, -4.0, 1)))

        selection = self.select(
            EncodedBranchModel(),
            {"clean": clean, "resize50": resized},
            candidate_weights=(0.0, 0.5, 1.0),
            batch_size=2,
        )

        self.assertEqual(selection.frequency_weight, 0.0)
        self.assertEqual(selection.worst_case_auroc, 1.0)
        self.assertEqual(selection.operation_aurocs, {"clean": 1.0, "resize50": 1.0})

    def test_rejects_invalid_selection_inputs(self):
        cases = (
            ({}, (0.0,), 1),
            ({"clean": BranchDataset(())}, (0.0,), 1),
            ({"clean": BranchDataset(((0, 0, 0),))}, (1.1,), 1),
            ({"clean": BranchDataset(((0, 0, 0),))}, (0.0,), 0),
        )
        for datasets, weights, batch_size in cases:
            with self.subTest(datasets=datasets, weights=weights, batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self.select(
                        EncodedBranchModel(),
                        datasets,
                        candidate_weights=weights,
                        batch_size=batch_size,
                    )


if __name__ == "__main__":
    unittest.main()
