"""Licensed image-dataset adapters for the optional neural training track."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from forensic_model.cifake import discover_cifake_candidates
from forensic_model.data_audit import AuditSummary, PartitionedSample, partition_candidates, sha256_file


@dataclass(frozen=True)
class DatasetAttribution:
    name: str
    source_url: str
    license_id: str
    citation: str


@dataclass(frozen=True)
class ImageExample:
    path: Path
    label: int
    content_group: str
    source: str
    generator_family: str


@dataclass(frozen=True)
class ImageDatasetBundle:
    train: tuple[ImageExample, ...]
    validation: tuple[ImageExample, ...]
    test: tuple[ImageExample, ...]
    audit: AuditSummary
    attributions: tuple[DatasetAttribution, ...]


def image_split_digest(bundle: ImageDatasetBundle) -> str:
    """Hash labels and content identities in every frozen split without paths."""

    digest = hashlib.sha256()
    for split, examples in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        for example in sorted(examples, key=lambda row: (row.content_group, row.label)):
            digest.update(f"{split}\0{example.content_group}\0{example.label}\n".encode("utf-8"))
    return digest.hexdigest()


CIFAKE_ATTRIBUTIONS = (
    DatasetAttribution(
        name="CIFAKE",
        source_url="https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images",
        license_id="MIT",
        citation="Bird, J. J. and Lotfi, A. (2024), CIFAKE, IEEE Access.",
    ),
    DatasetAttribution(
        name="CIFAR-10",
        source_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        license_id="MIT",
        citation="Krizhevsky, A. and Hinton, G. (2009), Learning multiple layers of features from tiny images.",
    ),
)

SYNTHSCARS_ATTRIBUTION = DatasetAttribution(
    name="SynthScars",
    source_url="https://huggingface.co/datasets/khr0516/SynthScars",
    license_id="Apache-2.0",
    citation="Kang, H. et al. (2025), LEGION: Learning to Ground and Explain for Synthetic Image Detection.",
)


class PillowImageDataset(Dataset[tuple[Tensor, int, str]]):
    def __init__(self, examples: Sequence[ImageExample], transform: Callable[[Image.Image], Tensor]) -> None:
        self.examples = tuple(examples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        example = self.examples[index]
        with Image.open(example.path) as source:
            image = source.convert("RGB")
        return self.transform(image), example.label, example.content_group


def discover_cifake(root: Path, *, seed: str = "cifake-split-v1") -> ImageDatasetBundle:
    """Audit every CIFAKE image and replace its leaky published split."""

    candidates = discover_cifake_candidates(root)
    partitioned, audit = partition_candidates(candidates, seed=seed)
    splits: dict[str, list[ImageExample]] = {"train": [], "validation": [], "test": []}
    for row in partitioned:
        splits[row.split].append(_cifake_example(row))
    return ImageDatasetBundle(
        train=tuple(splits["train"]),
        validation=tuple(splits["validation"]),
        test=tuple(splits["test"]),
        audit=audit,
        attributions=CIFAKE_ATTRIBUTIONS,
    )


def discover_synthscars_test(root: Path) -> tuple[ImageExample, ...]:
    """Load the held-out synthetic images without using its training split."""

    image_root = root / "test" / "images"
    if not image_root.is_dir():
        image_root = root / "SynthScars" / "test" / "images"
    examples = []
    for path in sorted(image_root.glob("*")):
        if path.is_file():
            digest = sha256_file(path)
            examples.append(ImageExample(path, 1, digest, "SynthScars", "unseen_mixed"))
    if not examples:
        raise ValueError("SynthScars test directory contains no images")
    return tuple(examples)


def training_transform(image_size: int = 32) -> Callable[[Image.Image], Tensor]:
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    return transforms.Compose(
        (
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.1, 0.03)], p=0.4),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.15),
            transforms.ToTensor(),
        )
    )


IMAGE_STRESS_OPERATIONS = (
    "clean",
    "jpeg30",
    "webp30",
    "resize50",
    "crop80",
    "blur1",
    "sharpen2",
    "noise02",
    "gamma08",
    "color70",
    "screenshot",
    "metadata_strip",
)


def evaluation_transform(image_size: int = 32, *, operation: str = "clean") -> Callable[[Image.Image], Tensor]:
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    if operation not in IMAGE_STRESS_OPERATIONS:
        raise ValueError(f"unsupported post-processing operation: {operation}")
    return transforms.Compose(
        (
            _Postprocess(operation),
            transforms.Resize(round(image_size * 1.125), antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        )
    )


class _Postprocess:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.operation == "clean":
            return image
        if self.operation == "jpeg30":
            return _codec_round_trip(image, "JPEG", quality=30)
        if self.operation == "webp30":
            return _codec_round_trip(image, "WEBP", quality=30, method=6)
        if self.operation == "blur1":
            return image.filter(ImageFilter.GaussianBlur(1.0))
        if self.operation == "resize50":
            half_size = (max(1, image.width // 2), max(1, image.height // 2))
            return image.resize(half_size, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        if self.operation == "crop80":
            margin_x = image.width // 10
            margin_y = image.height // 10
            cropped = image.crop((margin_x, margin_y, image.width - margin_x, image.height - margin_y))
            return cropped.resize(image.size, Image.Resampling.BICUBIC)
        if self.operation == "sharpen2":
            return ImageEnhance.Sharpness(image).enhance(2.0)
        if self.operation == "noise02":
            return _deterministic_noise(image, amplitude=0.02)
        if self.operation == "gamma08":
            lookup = [round(255.0 * ((value / 255.0) ** 0.8)) for value in range(256)]
            return image.point(lookup * 3)
        if self.operation == "color70":
            return ImageEnhance.Color(image).enhance(0.7)
        if self.operation == "screenshot":
            inner_size = (max(1, round(image.width * 0.88)), max(1, round(image.height * 0.88)))
            inner = image.resize(inner_size, Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", image.size, (238, 238, 238))
            canvas.paste(inner, ((image.width - inner.width) // 2, (image.height - inner.height) // 2))
            return _codec_round_trip(canvas, "JPEG", quality=85)
        if self.operation == "metadata_strip":
            return _codec_round_trip(image, "PNG", optimize=False)
        raise ValueError(f"unsupported post-processing operation: {self.operation}")


def _codec_round_trip(image: Image.Image, format_name: str, **options: int | bool) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format_name, **options)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _deterministic_noise(image: Image.Image, *, amplitude: float) -> Image.Image:
    maximum_delta = round(255 * amplitude)
    pixels = bytearray(image.tobytes())
    for index, value in enumerate(pixels):
        pseudo_random = ((index * 73 + index // 3 * 151 + 19) % 255) - 127
        delta = round(maximum_delta * pseudo_random / 127)
        pixels[index] = min(255, max(0, value + delta))
    return Image.frombytes("RGB", image.size, bytes(pixels))


def _cifake_example(row: PartitionedSample) -> ImageExample:
    synthetic = row.candidate.label == "synthetic"
    return ImageExample(
        path=row.candidate.path,
        label=int(synthetic),
        content_group=row.content_group,
        source="CIFAKE",
        generator_family="stable-diffusion-1.4" if synthetic else "",
    )
