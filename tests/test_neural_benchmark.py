import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "neural extra is not installed")
class NeuralImageBenchmarkTests(unittest.TestCase):
    def _fixture(self, root: Path):
        from PIL import Image

        from forensic_model.neural import NeuralConfig, create_scratch_detector, save_neural_checkpoint
        from forensic_model.neural_inference import CalibratedNeuralImageDetector

        checkpoint = root / "image.pt"
        images = root / "images"
        images.mkdir()
        for index in range(2):
            Image.new("RGB", (16, 16), (40 + index, 80, 120)).save(images / f"{index}.png")
        model = create_scratch_detector(
            NeuralConfig(spatial_widths=(4,), frequency_widths=(4,), dropout=0.0),
            seed=43,
        )
        save_neural_checkpoint(
            checkpoint,
            model,
            metadata={
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
                "threshold": 0.5,
                "frequency_weight": 0.0,
            },
        )
        return checkpoint, images, CalibratedNeuralImageDetector.load(checkpoint)

    def test_measures_tensor_file_and_memory_contracts(self) -> None:
        from forensic_model.neural_benchmark import ImageBenchmarkConfig, benchmark_image_inference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, images, detector = self._fixture(root)
            report = benchmark_image_inference(
                detector,
                sorted(images.glob("*.png")),
                checkpoint=checkpoint,
                config=ImageBenchmarkConfig(
                    image_size=16,
                    batch_size=2,
                    warmup_iterations=1,
                    tensor_iterations=2,
                    file_iterations=2,
                    cpu_threads=1,
                    seed=7,
                ),
            )

        self.assertEqual(report["tensor_inference"]["total_items"], 4)
        self.assertEqual(report["file_inference"]["total_items"], 2)
        self.assertGreater(report["memory"]["model_parameter_bytes"], 0)
        self.assertGreater(report["tensor_inference"]["items_per_second"], 0.0)

    def test_runner_writes_a_reusable_json_report(self) -> None:
        from forensic_model.neural_benchmark import ImageBenchmarkConfig, run_image_benchmark

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, images, _ = self._fixture(root)
            output = root / "report.json"
            run_image_benchmark(
                checkpoint,
                images,
                output=output,
                sample_count=1,
                config=ImageBenchmarkConfig(
                    image_size=16,
                    batch_size=1,
                    warmup_iterations=1,
                    tensor_iterations=1,
                    file_iterations=1,
                    cpu_threads=1,
                ),
            )
            parsed = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(parsed["sample_files"]), 1)
        self.assertEqual(parsed["measurement_scope"]["file_cache_state"], "warm operating-system file cache after explicit warmup")


if __name__ == "__main__":
    unittest.main()
