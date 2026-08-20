import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class CheckpointDigestTests(unittest.TestCase):
    def test_equivalent_serializations_have_one_semantic_digest(self) -> None:
        from forensic_model.checkpoint_digest import semantic_checkpoint_sha256

        payload = {
            "format": "example-v1",
            "config": {"width": 4},
            "metadata": {"seed": 17, "training": "scratch"},
            "state_dict": {
                "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "bias": torch.tensor(0.25),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            torch.save(payload, first)
            torch.save(payload, second)

            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                semantic_checkpoint_sha256(first),
                semantic_checkpoint_sha256(second),
            )

    def test_digest_changes_with_values_config_or_metadata(self) -> None:
        from forensic_model.checkpoint_digest import semantic_checkpoint_payload_sha256

        base = {
            "format": "example-v1",
            "config": {"width": 2},
            "metadata": {"seed": 3},
            "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        }
        changed_values = {**base, "state_dict": {"weight": torch.tensor([1.0, 2.1])}}
        changed_config = {**base, "config": {"width": 3}}
        changed_metadata = {**base, "metadata": {"seed": 4}}

        expected = semantic_checkpoint_payload_sha256(base)
        self.assertNotEqual(expected, semantic_checkpoint_payload_sha256(changed_values))
        self.assertNotEqual(expected, semantic_checkpoint_payload_sha256(changed_config))
        self.assertNotEqual(expected, semantic_checkpoint_payload_sha256(changed_metadata))

    def test_rejects_unsafe_or_incomplete_payload_shapes(self) -> None:
        from forensic_model.checkpoint_digest import semantic_checkpoint_payload_sha256

        with self.assertRaisesRegex(ValueError, "format"):
            semantic_checkpoint_payload_sha256({})
        with self.assertRaisesRegex(ValueError, "string names to tensors"):
            semantic_checkpoint_payload_sha256(
                {
                    "format": "example-v1",
                    "config": {},
                    "metadata": {},
                    "state_dict": {"weight": "not-a-tensor"},
                }
            )


if __name__ == "__main__":
    unittest.main()
