"""Deterministic local video decoding for optional neural temporal models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from forensic_model.neural_data import evaluation_transform
from forensic_model.neural_video_data import VideoExample


class VideoDecodeError(ValueError):
    """Raised when local media cannot produce a valid frame sequence."""


class VideoTensorDataset(Dataset[tuple[Tensor, int, str]]):
    """Decode each source into a fixed, uniformly sampled frame tensor."""

    def __init__(
        self,
        examples: Sequence[VideoExample],
        *,
        frame_count: int = 12,
        image_size: int = 64,
        operation: str = "clean",
        cache: bool = False,
    ) -> None:
        if frame_count < 3:
            raise ValueError("frame_count must be at least three")
        if image_size < 8:
            raise ValueError("image_size must be at least eight")
        self.examples = tuple(examples)
        self.frame_count = frame_count
        self.transform = evaluation_transform(image_size, operation=operation)
        self.cache = cache
        self._frame_cache: dict[int, Tensor] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        example = self.examples[index]
        cached = self._frame_cache.get(index)
        if cached is not None:
            return cached, example.label, example.content_group
        images = _read_images(example)
        chosen = uniform_indices(len(images), self.frame_count)
        frames = torch.stack(tuple(self.transform(images[position]) for position in chosen))
        if self.cache:
            self._frame_cache[index] = frames
        return frames, example.label, example.content_group


def uniform_indices(length: int, count: int) -> tuple[int, ...]:
    """Return deterministic endpoint-preserving indices, repeating short clips."""

    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if count == 1:
        return (0,)
    return tuple(round(step * (length - 1) / (count - 1)) for step in range(count))


def _read_images(example: VideoExample) -> tuple[Image.Image, ...]:
    try:
        if example.media_type == "frames":
            images = tuple(_open_image(path) for path in sorted(example.path.glob("*.jpg")))
        elif example.media_type == "video":
            images = _decode_video(example.path)
        else:
            raise VideoDecodeError(f"unsupported media type: {example.media_type}")
    except VideoDecodeError:
        raise
    except Exception as error:
        raise VideoDecodeError(f"failed to decode {example.path}: {type(error).__name__}") from error
    if len(images) < 2:
        raise VideoDecodeError(f"media contains fewer than two frames: {example.path}")
    return images


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _decode_video(path: Path) -> tuple[Image.Image, ...]:
    try:
        import av
    except ModuleNotFoundError as error:
        raise VideoDecodeError("video decoding requires the video extra") from error

    images = []
    with av.open(str(path)) as container:
        streams = container.streams.video
        if not streams:
            raise VideoDecodeError(f"media contains no video stream: {path}")
        for frame in container.decode(streams[0]):
            images.append(frame.to_image().convert("RGB"))
    return tuple(images)
