import csv
import tempfile
import unittest
from pathlib import Path

from forensic_model.manifest import ManifestError, Sample, load_manifest, manifest_digest, validate_manifest


def sample(**changes: str) -> Sample:
    values = {
        "sample_id": "real-1",
        "content_group": "scene-1",
        "path": "images/real-1.ppm",
        "sha256": "a" * 64,
        "label": "camera_or_human",
        "split": "train",
        "source": "fixture",
        "source_type": "procedural",
        "license_id": "CC0-1.0",
        "generator_family": "",
        "transformation": "original",
    }
    values.update(changes)
    return Sample(**values)


class ManifestValidationTests(unittest.TestCase):
    def test_accepts_complete_partitioned_rows(self) -> None:
        validate_manifest(
            [
                sample(),
                sample(
                    sample_id="synthetic-1",
                    content_group="scene-2",
                    sha256="b" * 64,
                    label="synthetic",
                    generator_family="fixture-a",
                ),
                sample(
                    sample_id="synthetic-holdout",
                    content_group="scene-3",
                    sha256="c" * 64,
                    label="synthetic",
                    split="generator_holdout",
                    generator_family="fixture-b",
                ),
            ]
        )

    def test_rejects_content_identity_leakage(self) -> None:
        rows = [sample(), sample(sample_id="real-2", sha256="b" * 64, split="test")]
        with self.assertRaisesRegex(ManifestError, "leaks across"):
            validate_manifest(rows)

    def test_rejects_generator_family_leakage(self) -> None:
        rows = [
            sample(label="synthetic", generator_family="family-a"),
            sample(
                sample_id="synthetic-2",
                content_group="scene-2",
                sha256="b" * 64,
                label="synthetic",
                split="generator_holdout",
                generator_family="FAMILY-A",
            ),
        ]
        with self.assertRaisesRegex(ManifestError, "generator families leak"):
            validate_manifest(rows)

    def test_rejects_unknown_license(self) -> None:
        with self.assertRaisesRegex(ManifestError, "license_id is not approved"):
            validate_manifest([sample(license_id="unknown")])

    def test_loads_csv_and_hashes_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sample().__dict__))
                writer.writeheader()
                writer.writerow(sample().__dict__)
            rows = load_manifest(path)
            self.assertEqual(rows[0].sample_id, "real-1")
            self.assertEqual(len(manifest_digest(path)), 64)


if __name__ == "__main__":
    unittest.main()
