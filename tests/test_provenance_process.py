import sys
import tempfile
import unittest
from pathlib import Path

from forensic_model.provenance import ProvenanceStatus, check_provenance
from forensic_model.provenance_process import JsonProcessVerifier, SignerTrustPolicy


VALID_SCRIPT = """
import hashlib, json, sys
path = sys.argv[-1]
digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
print(json.dumps({
    'status': 'valid',
    'media_sha256': digest,
    'credential_media_sha256': digest,
    'signer': 'trusted-publisher',
    'generator_asserted': True,
    'details': ['signature and asset binding verified'],
}))
"""


class ProcessVerifierTests(unittest.TestCase):
    def test_accepts_verified_result_from_trusted_signer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            verifier = JsonProcessVerifier(
                (sys.executable, "-c", VALID_SCRIPT),
                SignerTrustPolicy(frozenset({"trusted-publisher"})),
            )
            evidence = check_provenance(path, verifier)

        self.assertEqual(evidence.status, ProvenanceStatus.VALID)
        self.assertTrue(evidence.generator_asserted)

    def test_unlisted_signer_is_indeterminate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            verifier = JsonProcessVerifier((sys.executable, "-c", VALID_SCRIPT))
            evidence = check_provenance(path, verifier)

        self.assertEqual(evidence.status, ProvenanceStatus.INDETERMINATE)
        self.assertIn("not trusted", evidence.details[-1])

    def test_missing_verifier_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            evidence = check_provenance(path, JsonProcessVerifier(("/missing/verifier",)))

        self.assertEqual(evidence.status, ProvenanceStatus.UNSUPPORTED)

    def test_malformed_or_oversized_results_are_indeterminate(self) -> None:
        commands = (
            JsonProcessVerifier((sys.executable, "-c", "print('not json')")),
            JsonProcessVerifier(
                (sys.executable, "-c", "print('x' * 100)"),
                max_output_bytes=16,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            for verifier in commands:
                with self.subTest(command=verifier.command):
                    self.assertEqual(
                        check_provenance(path, verifier).status,
                        ProvenanceStatus.INDETERMINATE,
                    )

    def test_timeout_is_indeterminate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"media")
            verifier = JsonProcessVerifier(
                (sys.executable, "-c", "import time; time.sleep(1)"),
                timeout_seconds=0.01,
            )
            evidence = check_provenance(path, verifier)

        self.assertEqual(evidence.status, ProvenanceStatus.INDETERMINATE)
        self.assertIn("time limit", evidence.details[-1])


if __name__ == "__main__":
    unittest.main()
