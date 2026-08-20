import json
import tempfile
import unittest
from pathlib import Path

from forensic_model.data_audit import CandidateSample, partition_candidates
from forensic_model.identity_policy import apply_identity_policy


class IdentityPolicyTests(unittest.TestCase):
    def _candidate(self, sample_id: str, digest: str, label: str = "synthetic") -> CandidateSample:
        return CandidateSample(sample_id, Path(f"/{sample_id}.jpg"), label, digest * 64)

    def test_groups_observations_or_excludes_exact_hashes_explicitly(self) -> None:
        candidates = (
            self._candidate("red", "a"),
            self._candidate("blue", "b"),
            self._candidate("excluded", "c"),
        )
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema": "reviewed-collision-policy-v1",
                        "group_sha256": [["a" * 64, "b" * 64]],
                        "exclude_sha256": ["c" * 64],
                    }
                ),
                encoding="utf-8",
            )

            resolved, audit = apply_identity_policy(candidates, policy)
            partitioned, _ = partition_candidates(resolved, seed="split")

        self.assertEqual(len(resolved), 2)
        self.assertEqual(len({row.content_group for row in partitioned}), 1)
        self.assertEqual(len({row.split for row in partitioned}), 1)
        self.assertEqual(audit.reviewed_content_groups, 1)
        self.assertEqual(audit.grouped_observations, 2)
        self.assertEqual(audit.excluded_observations, 1)
        self.assertEqual(len(audit.policy_sha256), 64)

    def test_rejects_unknown_conflicting_or_cross_label_decisions(self) -> None:
        candidates = (
            self._candidate("real", "a", "camera_or_human"),
            self._candidate("fake", "b", "synthetic"),
            self._candidate("fake-two", "c", "synthetic"),
        )
        policies = (
            ({"group_sha256": [["a" * 64, "d" * 64]], "exclude_sha256": []}, "unknown hash"),
            ({"group_sha256": [["a" * 64, "b" * 64]], "exclude_sha256": []}, "one label"),
            (
                {"group_sha256": [["b" * 64, "c" * 64]], "exclude_sha256": ["b" * 64]},
                "one reviewed decision",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            for policy, message in policies:
                path.write_text(
                    json.dumps({"schema": "reviewed-collision-policy-v1", **policy}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    apply_identity_policy(candidates, path)


if __name__ == "__main__":
    unittest.main()
