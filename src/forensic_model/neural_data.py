"""Licensed image-dataset adapters for the optional neural training track."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from forensic_model.data_audit import AuditSummary, CandidateSample, PartitionedSample, partition_candidates, sha256_file


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

    candidates = []
    for path in sorted(root.glob("*/*/*.jpg")):
        relative = path.relative_to(root).as_posix()
        label = "synthetic" if path.parent.name.casefold() == "fake" else "camera_or_human"
        candidates.append(
            CandidateSample(
                sample_id=relative,
                path=path,
                label=label,
                sha256=sha256_file(path),
                perceptual_hash=_difference_hash(path),
            )
        )
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

    examples = []
    for path in sorted((root / "test" / "images").glob("*")):
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


def evaluation_transform(image_size: int = 32, *, operation: str = "clean") -> Callable[[Image.Image], Tensor]:
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    if operation not in {"clean", "jpeg30", "blur1", "resize50"}:
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
        if self.operation == "clean":
            return image
        if self.operation == "jpeg30":
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=30)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        if self.operation == "blur1":
            return image.filter(ImageFilter.GaussianBlur(1.0))
        half_size = (max(1, image.width // 2), max(1, image.height // 2))
        return image.resize(half_size, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)


def _difference_hash(path: Path) -> str:
    with Image.open(path) as source:
        values = list(source.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (values[row * 9 + column] > values[row * 9 + column + 1])
    return f"{bits:016x}"


def _cifake_example(row: PartitionedSample) -> ImageExample:
    synthetic = row.candidate.label == "synthetic"
    return ImageExample(
        path=row.candidate.path,
        label=int(synthetic),
        content_group=row.content_group,
        source="CIFAKE",
        generator_family="stable-diffusion-1.4" if synthetic else "",
    )
