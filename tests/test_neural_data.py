import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralDataTests(unittest.TestCase):
    def test_cifake_adapter_replaces_published_folders_with_hash_splits(self) -> None:
        from PIL import Image

        from forensic_model.neural_data import discover_cifake

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                for label in ("FAKE", "REAL"):
                    (root / split / label).mkdir(parents=True)
            rows = (
                ("train", "FAKE", 20),
                ("train", "REAL", 80),
                ("test", "FAKE", 140),
                ("test", "REAL", 220),
            )
            for index, (split, label, color) in enumerate(rows):
                Image.new("RGB", (12, 12), (color, color, color)).save(root / split / label / f"{index}.jpg")
            duplicate = root / "test" / "FAKE" / "duplicate.jpg"
            duplicate.write_bytes((root / "train" / "FAKE" / "0.jpg").read_bytes())

            bundle = discover_cifake(root, seed="fixture")

        self.assertEqual(bundle.audit.input_samples, 5)
        self.assertEqual(bundle.audit.retained_samples, 4)
        self.assertEqual(bundle.audit.exact_duplicates_removed, 1)
        all_rows = bundle.train + bundle.validation + bundle.test
        self.assertEqual({row.source for row in all_rows}, {"CIFAKE"})


if __name__ == "__main__":
    unittest.main()
