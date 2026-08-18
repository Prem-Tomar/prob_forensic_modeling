import tempfile
import unittest
from pathlib import Path

from forensic_model.neural_video_data import discover_video_benchmark


class VideoManifestTests(unittest.TestCase):
    def test_builds_balanced_source_grouped_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            davis, keling, sora = self._fixtures(root)

            first = discover_video_benchmark(davis, keling, sora)
            repeated = discover_video_benchmark(davis, keling, sora)

        self.assertEqual(first, repeated)
        self.assertEqual(first.audit.exclusions, 3)
        self.assertEqual(first.audit.duplicates_removed, 0)
        for examples in (first.train, first.validation, first.test):
            self.assertEqual({example.label for example in examples}, {0, 1})
        groups = [example.content_group for example in first.train + first.validation + first.test]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual({item.license_id for item in first.attributions}, {"CC-BY-NC-4.0"})

    def test_rejects_identical_content_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            davis, keling, sora = self._fixtures(root)
            training_video = next((keling / "keling" / "T2V").glob("*.mp4"))
            (sora / "duplicate.mp4").write_bytes(training_video.read_bytes())

            with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
                discover_video_benchmark(davis, keling, sora)

    def _fixtures(self, root: Path) -> tuple[Path, Path, Path]:
        davis = root / "DAVIS"
        manifests = davis / "ImageSets" / "2017"
        frames = davis / "JPEGImages" / "480p"
        manifests.mkdir(parents=True)
        train_names = [f"train-{index}" for index in range(8)]
        validation_names = [f"validation-{index}" for index in range(40)]
        (manifests / "train.txt").write_text("\n".join(train_names), encoding="utf-8")
        (manifests / "val.txt").write_text("\n".join(validation_names), encoding="utf-8")
        for index, name in enumerate(train_names + validation_names):
            sequence = frames / name
            sequence.mkdir(parents=True)
            (sequence / "00000.jpg").write_bytes(f"real-{index}".encode())

        videos = root / "Keling" / "keling" / "T2V"
        videos.mkdir(parents=True)
        for index in range(40):
            (videos / f"generated-{index}.mp4").write_bytes(f"keling-{index}".encode())
        for name in ("39462_1717765170_raw.mp4", "WeChat_20240608171016.mp4"):
            (videos / name).write_bytes(name.encode())

        sora = root / "Sora"
        sora.mkdir()
        for index in range(8):
            (sora / f"sora-{index}.mp4").write_bytes(f"sora-{index}".encode())
        (sora / "Sora | OpenAI.mp4").write_bytes(b"excluded")
        return davis, root / "Keling", sora


if __name__ == "__main__":
    unittest.main()
