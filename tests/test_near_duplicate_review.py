import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


@unittest.skipUnless(Image is not None, "common image extra is not installed")
class NearDuplicateReviewTests(unittest.TestCase):
    def test_cli_writes_report_and_optional_contact_sheet(self) -> None:
        from forensic_model.data_audit import CandidateSample
        from forensic_model.near_duplicate_review import main

        candidates = (
            CandidateSample("one", Path("/one.jpg"), "synthetic", "a" * 64, "abcd"),
            CandidateSample("two", Path("/two.jpg"), "synthetic", "b" * 64, "abcd"),
        )
        report = {"groups": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            contact = root / "contact.png"
            with (
                patch(
                    "forensic_model.near_duplicate_review.discover_cifake_candidates",
                    return_value=candidates,
                ),
                patch(
                    "forensic_model.near_duplicate_review.review_perceptual_collisions",
                    return_value=report,
                ) as review,
                patch("forensic_model.near_duplicate_review.write_contact_sheet") as sheet,
            ):
                status = main(
                    [
                        "--cifake-root",
                        "/data/cifake",
                        "--output",
                        str(output),
                        "--contact-sheet",
                        str(contact),
                        "--thumbnail-size",
                        "32",
                        "--low-distance-threshold",
                        "0.05",
                    ]
                )

        self.assertEqual(status, 0)
        review.assert_called_once_with(candidates, thumbnail_size=32, low_distance_threshold=0.05)
        sheet.assert_called_once_with(candidates, report, output=contact)

    def test_collision_review_measures_without_merging(self) -> None:
        from forensic_model.data_audit import CandidateSample, sha256_file
        from forensic_model.near_duplicate_review import review_perceptual_collisions

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            unrelated = root / "unrelated.png"
            Image.new("RGB", (16, 16), (100, 120, 140)).save(first)
            Image.new("RGB", (16, 16), (101, 121, 141)).save(second)
            Image.new("RGB", (16, 16), (240, 10, 20)).save(unrelated)
            rows = (
                CandidateSample("first", first, "synthetic", sha256_file(first), "abcd"),
                CandidateSample("second", second, "synthetic", sha256_file(second), "abcd"),
                CandidateSample("third", unrelated, "camera_or_human", sha256_file(unrelated), "ffff"),
            )

            report = review_perceptual_collisions(rows)

        self.assertEqual(report["collision_group_count"], 1)
        self.assertEqual(report["pair_comparison_count"], 1)
        self.assertEqual(report["low_pixel_distance_pair_count"], 1)
        self.assertEqual(report["cross_label_pair_count"], 0)
        self.assertEqual(report["groups"][0]["member_count"], 2)

    def test_review_detects_cross_label_hash_collision(self) -> None:
        from forensic_model.data_audit import CandidateSample, sha256_file
        from forensic_model.near_duplicate_review import review_perceptual_collisions

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (8, 8), "black").save(first)
            Image.new("RGB", (8, 8), "white").save(second)
            rows = (
                CandidateSample("first", first, "synthetic", sha256_file(first), "0000"),
                CandidateSample("second", second, "camera_or_human", sha256_file(second), "0000"),
            )

            report = review_perceptual_collisions(rows)

        self.assertEqual(report["cross_label_pair_count"], 1)
        comparison = report["groups"][0]["comparisons"][0]
        self.assertEqual(comparison["automated_screen"], "hash_collision_only")
        self.assertAlmostEqual(comparison["grayscale_mean_absolute_error"], 1.0)

    def test_invalid_review_threshold_is_rejected(self) -> None:
        from forensic_model.near_duplicate_review import review_perceptual_collisions

        with self.assertRaisesRegex(ValueError, "threshold"):
            review_perceptual_collisions((), low_distance_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
