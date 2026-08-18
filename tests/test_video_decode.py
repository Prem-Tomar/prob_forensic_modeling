import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from PIL import Image
except ModuleNotFoundError:
    torch = None
    Image = None


@unittest.skipUnless(torch is not None, "video extra is not installed")
class VideoDecodeTests(unittest.TestCase):
    def test_decodes_frame_directory_to_fixed_tensor(self) -> None:
        from forensic_model.neural_video_data import VideoExample
        from forensic_model.video_decode import VideoTensorDataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new("RGB", (16, 12), (0, 0, 0)).save(path / "00000.jpg")
            Image.new("RGB", (16, 12), (255, 255, 255)).save(path / "00001.jpg")
            example = VideoExample(path, 0, "group", "fixture", "", "train", "frames")

            frames, label, group = VideoTensorDataset((example,), frame_count=4, image_size=8)[0]

        self.assertEqual(tuple(frames.shape), (4, 3, 8, 8))
        self.assertEqual(label, 0)
        self.assertEqual(group, "group")
        torch.testing.assert_close(frames[0], frames[1])
        torch.testing.assert_close(frames[2], frames[3])
        self.assertGreater(float(frames[3].mean()), float(frames[0].mean()))

    def test_uniform_sampling_preserves_endpoints(self) -> None:
        from forensic_model.video_decode import uniform_indices

        self.assertEqual(uniform_indices(2, 4), (0, 0, 1, 1))
        self.assertEqual(uniform_indices(10, 3), (0, 4, 9))
        with self.assertRaises(ValueError):
            uniform_indices(0, 3)


if __name__ == "__main__":
    unittest.main()
