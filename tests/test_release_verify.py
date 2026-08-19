import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forensic_model.release_verify import verify_release_wheel


class ReleaseVerificationTests(unittest.TestCase):
    def test_records_byte_identical_isolated_smoke_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "example.whl"
            expected = root / "expected.json"
            output = root / "verification.json"
            wheel.write_bytes(b"wheel")
            expected.write_bytes(b'{"result": "ok"}\n')

            def fake_run(command, **options):
                self.assertTrue(options["check"])
                self.assertIn(subprocess.PIPE, (options["stdout"], options["stderr"]))
                if "smoke-evaluate" in command:
                    Path(command[-1]).write_bytes(expected.read_bytes())

            import subprocess

            with patch("forensic_model.release_verify.venv.EnvBuilder.create"), patch(
                "forensic_model.release_verify.subprocess.run", side_effect=fake_run
            ):
                verification = verify_release_wheel(wheel, expected, output=output)

            parsed = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(verification.smoke_report_byte_identical)
        self.assertEqual(parsed["expected_smoke_sha256"], parsed["installed_smoke_sha256"])
        self.assertEqual(parsed["install_mode"], "local_wheel_no_index_no_dependencies")

    def test_rejects_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "existing wheel"):
                verify_release_wheel(root / "missing.whl", root / "missing.json", output=root / "out.json")


if __name__ == "__main__":
    unittest.main()
