import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from forensic_model.manifest import REQUIRED_COLUMNS
from forensic_model.manifest_video import load_manifest_video_corpus


class ManifestVideoCorpusTests(unittest.TestCase):
    def test_loads_attributed_splits_and_matched_generator_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self._row(root, "train-real", "train", "camera_or_human", "", "training"),
                self._row(root, "train-fake", "train", "synthetic", "generator-a", "training"),
                self._row(root, "holdout-real", "generator_holdout", "camera_or_human", "", "holdout"),
                self._row(
                    root,
                    "holdout-fake",
                    "generator_holdout",
                    "synthetic",
                    "Generator-B",
                    "holdout",
                ),
            ]
            rows[2]["content_group"] = rows[3]["content_group"] = "shared-prompt"
            manifest = self._manifest(root, rows)

            corpus = load_manifest_video_corpus(manifest, media_root=root)
            matched = corpus.matched_generator_holdouts()["generator-b"]

        self.assertEqual(len(corpus.examples("train")), 2)
        self.assertEqual(corpus.audit.verified_files, 4)
        self.assertEqual(len(corpus.audit.attributions), 2)
        self.assertEqual([row.label for row in matched.examples], [0, 1])
        self.assertEqual({row.media_type for row in matched.examples}, {"video"})

    def test_rejects_tampering_unsupported_media_and_unmatched_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = self._row(
                root,
                "generated",
                "generator_holdout",
                "synthetic",
                "generator-b",
                "holdout",
            )
            manifest = self._manifest(root, [generated])
            corpus = load_manifest_video_corpus(manifest, media_root=root)
            with self.assertRaisesRegex(ValueError, "requires one real and one synthetic"):
                corpus.matched_generator_holdouts()

            (root / generated["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest_video_corpus(manifest, media_root=root)

            unsupported = dict(generated)
            unsupported["sample_id"] = "unsupported"
            unsupported["path"] = "media/unsupported.txt"
            path = root / unsupported["path"]
            path.write_bytes(b"text")
            unsupported["sha256"] = hashlib.sha256(b"text").hexdigest()
            with self.assertRaisesRegex(ValueError, "not a supported video"):
                load_manifest_video_corpus(self._manifest(root, [unsupported]), media_root=root)

    def test_rejects_conflicting_matched_semantic_categories(self) -> None:
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
            corpus = load_manifest_video_corpus(self._manifest(root, [real, generated]), media_root=root)
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
        relative = Path("media") / f"{sample_id}.mp4"
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
        path = root / "video-manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path


if __name__ == "__main__":
    unittest.main()
