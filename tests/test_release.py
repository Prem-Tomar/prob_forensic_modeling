import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forensic_model.release import build_release_wheel


class ReleaseBuildTests(unittest.TestCase):
    def test_stages_only_package_inputs_and_hashes_one_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "dist"
            manifest = root / "release.json"
            (source / "src" / "example").mkdir(parents=True)
            (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
            (source / "README.md").write_text("example\n", encoding="utf-8")
            (source / "src" / "example" / "__init__.py").write_text("", encoding="utf-8")
            (source / "requirements-build.lock").write_text("build==1\n", encoding="utf-8")

            def fake_build(command, **options):
                self.assertFalse(options.get("shell", False))
                self.assertTrue(options["check"])
                self.assertEqual(options["env"]["SOURCE_DATE_EPOCH"], "315532800")
                staged = Path(command[-1])
                self.assertTrue((staged / "src" / "example" / "__init__.py").is_file())
                build_output = Path(command[command.index("--outdir") + 1])
                build_output.mkdir()
                (build_output / "example-0.1-py3-none-any.whl").write_bytes(b"wheel")

            with patch("forensic_model.release.subprocess.run", side_effect=fake_build):
                artifact = build_release_wheel(source, output, manifest=manifest)

            parsed = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(artifact.filename, "example-0.1-py3-none-any.whl")
        self.assertEqual(artifact.size_bytes, 5)
        self.assertEqual(artifact.source_date_epoch, 315532800)
        self.assertEqual(parsed["sha256"], artifact.sha256)
        self.assertIsNotNone(parsed["build_requirements_sha256"])

    def test_rejects_incomplete_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "source root"):
                build_release_wheel(Path(directory), Path(directory) / "dist")


if __name__ == "__main__":
    unittest.main()
