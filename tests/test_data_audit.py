import tempfile
import unittest
from pathlib import Path

from forensic_model.data_audit import CandidateSample, DataAuditError, partition_candidates, sha256_file


def candidate(
    sample_id: str,
    digest: str,
    label: str = "synthetic",
    perceptual_hash: str = "",
    reviewed_content_group: str = "",
) -> CandidateSample:
    return CandidateSample(
        sample_id,
        Path(f"/{sample_id}.jpg"),
        label,
        digest * 64,
        perceptual_hash,
        reviewed_content_group,
    )


class DataAuditTests(unittest.TestCase):
    def test_file_hash_reads_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"forensic sample")
            self.assertEqual(
                sha256_file(path),
                "48dcbacaa277dae675ea434f0d94c862e161d8646c919d19c13543676701cf5b",
            )

    def test_partition_is_order_independent_and_removes_exact_duplicates(self) -> None:
        rows = [candidate("b", "a"), candidate("duplicate", "a"), candidate("c", "b", "camera_or_human")]
        first, summary = partition_candidates(rows, seed="published-split-v1")
        second, _ = partition_candidates(reversed(rows), seed="published-split-v1")

        self.assertEqual([(row.content_group, row.split) for row in first], [(row.content_group, row.split) for row in second])
        self.assertEqual(summary.input_samples, 3)
        self.assertEqual(summary.retained_samples, 2)
        self.assertEqual(summary.exact_duplicate_groups, 1)
        self.assertEqual(summary.exact_duplicates_removed, 1)

    def test_perceptual_collisions_are_reported_without_silent_merging(self) -> None:
        rows = [candidate("one", "a", perceptual_hash="f0"), candidate("two", "b", perceptual_hash="f0")]
        retained, summary = partition_candidates(rows, seed="split")
        self.assertEqual(len(retained), 2)
        self.assertEqual(summary.perceptual_collision_groups, 1)

    def test_conflicting_duplicate_labels_are_rejected(self) -> None:
        rows = [candidate("real", "a", "camera_or_human"), candidate("fake", "a", "synthetic")]
        with self.assertRaisesRegex(DataAuditError, "conflicting labels"):
            partition_candidates(rows, seed="split")

    def test_reviewed_identity_keeps_observations_in_one_partition(self) -> None:
        rows = [
            candidate("red", "a", reviewed_content_group="reviewed-scene"),
            candidate("blue", "b", reviewed_content_group="reviewed-scene"),
        ]

        retained, summary = partition_candidates(rows, seed="split")

        self.assertEqual(len(retained), 2)
        self.assertEqual({row.content_group for row in retained}, {"reviewed-scene"})
        self.assertEqual(len({row.split for row in retained}), 1)
        self.assertEqual(summary.retained_samples, 2)

    def test_reviewed_identity_rejects_conflicting_labels(self) -> None:
        rows = [
            candidate("real", "a", "camera_or_human", reviewed_content_group="reviewed-scene"),
            candidate("fake", "b", "synthetic", reviewed_content_group="reviewed-scene"),
        ]
        with self.assertRaisesRegex(DataAuditError, "reviewed content group.*conflicting labels"):
            partition_candidates(rows, seed="split")


if __name__ == "__main__":
    unittest.main()
