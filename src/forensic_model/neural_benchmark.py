"""Local latency and memory measurement for calibrated neural image inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from forensic_model.neural_inference import CalibratedNeuralImageDetector


@dataclass(frozen=True)
class ImageBenchmarkConfig:
    image_size: int = 32
    batch_size: int = 64
    warmup_iterations: int = 5
    tensor_iterations: int = 50
    file_iterations: int = 64
    cpu_threads: int = 8
    seed: int = 20260818

    def __post_init__(self) -> None:
        counts = (
            self.image_size,
            self.batch_size,
            self.warmup_iterations,
            self.tensor_iterations,
            self.file_iterations,
            self.cpu_threads,
        )
        if self.image_size < 8 or any(value <= 0 for value in counts[1:]) or self.seed < 0:
            raise ValueError("benchmark sizes and counts must be positive and seed must be non-negative")


def benchmark_image_inference(
    detector: CalibratedNeuralImageDetector,
    image_paths: Sequence[Path],
    *,
    checkpoint: Path,
    config: ImageBenchmarkConfig = ImageBenchmarkConfig(),
) -> dict[str, object]:
    """Measure tensor inference and local file decode-to-decision latency."""

    paths = tuple(image_paths)
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("benchmark requires existing local image files")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        generator = torch.Generator().manual_seed(config.seed)
        batch = torch.rand(
            config.batch_size,
            3,
            config.image_size,
            config.image_size,
            generator=generator,
        )
        for _ in range(config.warmup_iterations):
            detector.predict_tensors(batch)
        tensor_durations = []
        for _ in range(config.tensor_iterations):
            started = time.perf_counter_ns()
            detector.predict_tensors(batch)
            tensor_durations.append((time.perf_counter_ns() - started) / 1_000_000_000.0)

        for index in range(config.warmup_iterations):
            detector.predict_file(paths[index % len(paths)], image_size=config.image_size)
        file_durations = []
        for index in range(config.file_iterations):
            started = time.perf_counter_ns()
            detector.predict_file(paths[index % len(paths)], image_size=config.image_size)
            file_durations.append((time.perf_counter_ns() - started) / 1_000_000_000.0)
    finally:
        torch.set_num_threads(previous_threads)

    parameter_bytes = sum(value.numel() * value.element_size() for value in detector.model.parameters())
    buffer_bytes = sum(value.numel() * value.element_size() for value in detector.model.buffers())
    return {
        "claim_scope": "local_cpu_measurement_not_capacity_planning",
        "configuration": asdict(config),
        "environment": {
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "checkpoint_sha256": _sha256(checkpoint),
        "measurement_scope": {
            "tensor": "calibrated batch decision from an in-memory float32 RGB tensor",
            "file": "local file open, RGB conversion, deterministic resize, calibrated decision",
            "file_cache_state": "warm operating-system file cache after explicit warmup",
        },
        "tensor_inference": _latency_summary(tensor_durations, items_per_iteration=config.batch_size),
        "file_inference": _latency_summary(file_durations, items_per_iteration=1),
        "memory": {
            "checkpoint_file_bytes": checkpoint.stat().st_size,
            "model_parameter_bytes": parameter_bytes,
            "model_buffer_bytes": buffer_bytes,
            "input_batch_bytes": batch.numel() * batch.element_size(),
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
        "sample_files": [str(path) for path in paths],
    }


def run_image_benchmark(
    checkpoint: Path,
    image_root: Path,
    *,
    output: Path,
    sample_count: int = 32,
    config: ImageBenchmarkConfig = ImageBenchmarkConfig(),
) -> dict[str, object]:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    paths = _discover_images(image_root)[:sample_count]
    if not paths:
        raise ValueError("image benchmark root contains no supported images")
    report = benchmark_image_inference(
        CalibratedNeuralImageDetector.load(checkpoint),
        paths,
        checkpoint=checkpoint,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-benchmark-image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--tensor-iterations", type=int, default=50)
    parser.add_argument("--file-iterations", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_image_benchmark(
        options.checkpoint,
        options.image_root,
        output=options.output,
        sample_count=options.sample_count,
        config=ImageBenchmarkConfig(
            image_size=options.image_size,
            batch_size=options.batch_size,
            warmup_iterations=options.warmup_iterations,
            tensor_iterations=options.tensor_iterations,
            file_iterations=options.file_iterations,
            cpu_threads=options.cpu_threads,
            seed=options.seed,
        ),
    )
    return 0


def _latency_summary(durations: Sequence[float], *, items_per_iteration: int) -> dict[str, float | int]:
    ordered = sorted(durations)
    total_items = len(ordered) * items_per_iteration
    total_seconds = sum(ordered)
    return {
        "iterations": len(ordered),
        "items_per_iteration": items_per_iteration,
        "total_items": total_items,
        "total_seconds": total_seconds,
        "median_iteration_milliseconds": statistics.median(ordered) * 1000.0,
        "p95_iteration_milliseconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] * 1000.0,
        "mean_milliseconds_per_item": total_seconds * 1000.0 / total_items,
        "items_per_second": total_items / total_seconds,
    }


def _discover_images(root: Path) -> tuple[Path, ...]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return tuple(path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.casefold() in suffixes)


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ModuleNotFoundError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
