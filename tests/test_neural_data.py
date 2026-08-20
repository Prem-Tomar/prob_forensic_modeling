import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralDataTests(unittest.TestCase):
    def test_stress_operations_are_complete_deterministic_and_shape_stable(self) -> None:
        from PIL import Image

        from forensic_model.neural_data import IMAGE_STRESS_OPERATIONS, evaluation_transform

        self.assertEqual(
            set(IMAGE_STRESS_OPERATIONS),
            {
                "clean",
                "jpeg30",
                "webp30",
                "resize50",
                "crop80",
                "blur1",
                "sharpen2",
                "noise02",
                "gamma08",
                "color70",
                "screenshot",
                "metadata_strip",
            },
        )
        image = Image.new("RGB", (40, 36))
        image.putdata([(index * 7 % 256, index * 13 % 256, index * 29 % 256) for index in range(40 * 36)])
        for operation in IMAGE_STRESS_OPERATIONS:
            transform = evaluation_transform(operation=operation)
            first = transform(image)
            second = transform(image)
            self.assertEqual(tuple(first.shape), (3, 32, 32), operation)
            self.assertTrue(torch.equal(first, second), operation)

    def test_metadata_strip_is_pixel_preserving(self) -> None:
        from PIL import Image

        from forensic_model.neural_data import _Postprocess

        image = Image.new("RGB", (11, 9), (20, 40, 60))
        stripped = _Postprocess("metadata_strip")(image)

        self.assertEqual(stripped.tobytes(), image.tobytes())

    def test_unknown_stress_operation_is_rejected(self) -> None:
        from forensic_model.neural_data import evaluation_transform

        with self.assertRaisesRegex(ValueError, "unsupported post-processing"):
            evaluation_transform(operation="unknown")

    def test_cifake_adapter_replaces_published_folders_with_hash_splits(self) -> None:
        from PIL import Image

        from forensic_model.neural_data import discover_cifake, image_split_digest

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
            repeated = discover_cifake(root, seed="fixture")

        self.assertEqual(bundle.audit.input_samples, 5)
        self.assertEqual(bundle.audit.retained_samples, 4)
        self.assertEqual(bundle.audit.exact_duplicates_removed, 1)
        all_rows = bundle.train + bundle.validation + bundle.test
        self.assertEqual({row.source for row in all_rows}, {"CIFAKE"})
        self.assertEqual(image_split_digest(bundle), image_split_digest(repeated))

    def test_synthscars_adapter_accepts_archive_wrapper_directory(self) -> None:
        from PIL import Image

        from forensic_model.neural_data import discover_synthscars_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "SynthScars" / "test" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (12, 12), (40, 80, 120)).save(images / "sample.png")

            rows = discover_synthscars_test(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "SynthScars")
        self.assertEqual(rows[0].label, 1)

    def test_cifake_adapter_applies_explicit_reviewed_identity_policy(self) -> None:
        from PIL import Image

        from forensic_model.data_audit import sha256_file
        from forensic_model.neural_data import discover_cifake

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "train" / "REAL"
            images.mkdir(parents=True)
            first = images / "red.jpg"
            second = images / "blue.jpg"
            Image.new("RGB", (12, 12), "red").save(first)
            Image.new("RGB", (12, 12), "blue").save(second)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema": "reviewed-collision-policy-v1",
                        "group_sha256": [[sha256_file(first), sha256_file(second)]],
                        "exclude_sha256": [],
                    }
                ),
                encoding="utf-8",
            )

            bundle = discover_cifake(root, seed="fixture", identity_policy=policy)

        all_rows = bundle.train + bundle.validation + bundle.test
        self.assertEqual(len({row.content_group for row in all_rows}), 1)
        self.assertEqual(sum(bool(split) for split in (bundle.train, bundle.validation, bundle.test)), 1)
        self.assertIsNotNone(bundle.identity_policy)
        self.assertEqual(bundle.identity_policy.reviewed_content_groups, 1)


if __name__ == "__main__":
    unittest.main()
