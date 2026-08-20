import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class ManifestHoldoutScoringTests(unittest.TestCase):
    def test_scores_calibrated_probabilities_without_reordering_groups(self) -> None:
        from forensic_model.manifest_holdout_experiment import _score

        class Detector:
            def predict_tensors(self, images):
                return tuple(
                    SimpleNamespace(probability_synthetic=float(image.mean()))
                    for image in images
                )

        rows = [
            (torch.zeros(3, 8, 8), 0, "content-a"),
            (torch.ones(3, 8, 8), 1, "content-a"),
            (torch.full((3, 8, 8), 0.25), 0, "content-b"),
            (torch.full((3, 8, 8), 0.75), 1, "content-b"),
        ]

        probabilities, labels, groups = _score(Detector(), rows, batch_size=2)

        self.assertEqual(probabilities, [0.0, 1.0, 0.25, 0.75])
        self.assertEqual(labels, [0, 1, 0, 1])
        self.assertEqual(groups, ["content-a", "content-a", "content-b", "content-b"])

    def test_runs_path_free_ineligible_family_report(self) -> None:
        from PIL import Image

        from forensic_model.manifest import REQUIRED_COLUMNS
        from forensic_model.manifest_holdout_experiment import run_manifest_holdout_experiment

        class Detector:
            threshold = 0.5
            abstain_margin = 0.05

            def predict_tensors(self, images):
                return tuple(SimpleNamespace(probability_synthetic=float(image.mean())) for image in images)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                for label, value in (("camera_or_human", index), ("synthetic", 255 - index)):
                    sample_id = f"{label}-{index}"
                    relative = Path("media") / f"{sample_id}.png"
                    path = root / relative
                    path.parent.mkdir(exist_ok=True)
                    Image.new("RGB", (8, 8), (value, value, value)).save(path)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "content_group": f"prompt-{index}",
                            "path": relative.as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "label": label,
                            "split": "generator_holdout",
                            "source": "fixture",
                            "source_url": "https://example.test/fixture",
                            "source_type": "generated" if label == "synthetic" else "camera",
                            "license_id": "MIT",
                            "citation": "Fixture dataset, version 1.",
                            "generator_family": "generator-b" if label == "synthetic" else "",
                            "transformation": "original",
                            "semantic_category": "bright" if index == 0 else "dark",
                            "capture_device": "camera-a" if label == "camera_or_human" else "",
                        }
                    )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "format": "fixture-v1",
                    "config": {},
                    "metadata": {"seed": 1},
                    "state_dict": {"weight": torch.tensor([1.0])},
                },
                checkpoint,
            )
            output = root / "report.json"

            with patch(
                "forensic_model.manifest_holdout_experiment.CalibratedNeuralImageDetector.load",
                return_value=Detector(),
            ):
                report = run_manifest_holdout_experiment(
                    manifest,
                    root,
                    checkpoint,
                    output=output,
                    image_size=8,
                    batch_size=2,
                    bootstrap_resamples=20,
                )
            rendered = output.read_text(encoding="utf-8")
            parsed = json.loads(rendered)

        self.assertEqual(report["aggregate"]["macro_auroc"], 1.0)
        self.assertFalse(report["acceptance"]["eligible"])
        self.assertFalse(report["acceptance"]["all_gates_passed"])
        self.assertEqual(parsed["generator_families"]["generator-b"]["metrics"]["count"], 4)
        self.assertEqual(
            parsed["generator_families"]["generator-b"]["diagnostics"]["selective"]["count"],
            4,
        )
        self.assertTrue(parsed["generator_families"]["generator-b"]["slice_coverage"]["metadata_complete"])
        self.assertEqual(
            parsed["generator_families"]["generator-b"]["slices"]["capture_device"]["camera-a"]["status"],
            "evaluated",
        )
        self.assertEqual(
            parsed["generator_families"]["generator-b"]["slices"]["semantic_category"]["bright"]["status"],
            "insufficient_group_coverage",
        )
        self.assertNotIn(str(root), rendered)


if __name__ == "__main__":
    unittest.main()
