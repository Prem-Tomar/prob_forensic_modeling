import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

from forensic_model.manifest import REQUIRED_COLUMNS


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class ManifestImageCorpusTests(unittest.TestCase):
    def test_loads_verified_portable_splits_and_generator_holdouts(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self._row(root, "train-real", "train", "camera_or_human", "", "real-source"),
                self._row(root, "train-fake", "train", "synthetic", "generator-a", "generated-source"),
                self._row(root, "validation-real", "validation", "camera_or_human", "", "real-source"),
                self._row(
                    root,
                    "holdout-fake",
                    "generator_holdout",
                    "synthetic",
                    "Generator-B",
                    "holdout-source",
                ),
            ]
            manifest = self._manifest(root, rows)

            corpus = load_manifest_image_corpus(manifest, media_root=root)

        self.assertEqual(len(corpus.examples("train")), 2)
        self.assertEqual(corpus.audit.verified_files, 4)
        self.assertEqual(corpus.audit.generator_counts, {"generator-a": 1, "generator-b": 1})
        self.assertEqual({item.source for item in corpus.audit.attributions}, {"generated-source", "holdout-source", "real-source"})
        self.assertEqual(tuple(corpus.generator_holdouts()), ("generator-b",))
        self.assertEqual(corpus.generator_holdouts()["generator-b"][0].label, 1)

    def test_rejects_media_tampering_and_paths_outside_root(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._row(root, "sample", "train", "camera_or_human", "", "source")
            manifest = self._manifest(root, [row])
            (root / row["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest_image_corpus(manifest, media_root=root)

            outside = root.parent / f"{root.name}-outside.jpg"
            escaped = dict(row)
            escaped["path"] = f"../{outside.name}"
            escaped["sha256"] = hashlib.sha256(b"outside").hexdigest()
            outside.write_bytes(b"outside")
            manifest = self._manifest(root, [escaped])
            try:
                with self.assertRaisesRegex(ValueError, "escapes media root"):
                    load_manifest_image_corpus(manifest, media_root=root)
            finally:
                outside.unlink(missing_ok=True)

    def test_rejects_unknown_split_access(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._row(root, "sample", "train", "camera_or_human", "", "source")
            corpus = load_manifest_image_corpus(self._manifest(root, [row]), media_root=root)
            with self.assertRaisesRegex(ValueError, "does not contain split"):
                corpus.examples("test")

    def test_builds_one_to_one_content_matched_generator_slices(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = self._row(root, "real", "generator_holdout", "camera_or_human", "", "camera-source")
            generated = self._row(
                root,
                "generated",
                "generator_holdout",
                "synthetic",
                "generator-b",
                "generated-source",
            )
            real["content_group"] = generated["content_group"] = "shared-prompt"
            corpus = load_manifest_image_corpus(self._manifest(root, [real, generated]), media_root=root)

            matched = corpus.matched_generator_holdouts()["generator-b"]

        self.assertEqual(len(matched.real_controls), 1)
        self.assertEqual(len(matched.synthetic_examples), 1)
        self.assertEqual([row.label for row in matched.examples], [0, 1])

    def test_rejects_unmatched_generator_slice(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = self._row(
                root,
                "generated",
                "generator_holdout",
                "synthetic",
                "generator-b",
                "generated-source",
            )
            corpus = load_manifest_image_corpus(self._manifest(root, [generated]), media_root=root)
            with self.assertRaisesRegex(ValueError, "requires one real and one synthetic"):
                corpus.matched_generator_holdouts()

    def test_rejects_conflicting_matched_semantic_categories(self) -> None:
        from forensic_model.manifest_image import load_manifest_image_corpus

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = self._row(root, "real", "generator_holdout", "camera_or_human", "", "camera-source")
            generated = self._row(
                root,
                "generated",
                "generator_holdout",
                "synthetic",
                "generator-b",
                "generated-source",
            )
            real["content_group"] = generated["content_group"] = "shared-prompt"
            generated["semantic_category"] = "portrait"
            corpus = load_manifest_image_corpus(self._manifest(root, [real, generated]), media_root=root)
            with self.assertRaisesRegex(ValueError, "conflicting semantic_category"):
                corpus.matched_generator_holdouts()

    def _row(
        self,
        root: Path,
        sample_id: str,
        split: str,
        label: str,
        generator_family: str,
        source: str,
    ) -> dict[str, str]:
        relative = Path("media") / f"{sample_id}.jpg"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        content = sample_id.encode("utf-8")
        path.write_bytes(content)
        return {
            "sample_id": sample_id,
            "content_group": f"content-{sample_id}",
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "label": label,
            "split": split,
            "source": source,
            "source_url": f"https://example.test/{source}",
            "source_type": "generated" if label == "synthetic" else "camera",
            "license_id": "MIT",
            "citation": f"{source} fixture dataset.",
            "generator_family": generator_family,
            "transformation": "original",
            "semantic_category": "landscape",
            "capture_device": "fixture-camera" if label == "camera_or_human" else "",
        }

    def _manifest(self, root: Path, rows: list[dict[str, str]]) -> Path:
        path = root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path


if __name__ == "__main__":
    unittest.main()
