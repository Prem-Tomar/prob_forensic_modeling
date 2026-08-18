"""Image containers and decoder adapters used by the public library."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded safely as a supported image."""


MAX_PIXELS = 40_000_000


@dataclass(frozen=True)
class RGBImage:
    """Immutable row-major RGB pixels with normalized channel values."""

    width: int
    height: int
    pixels: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("images must be at least 2 by 2 pixels")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("pixel count does not match dimensions")
        if any(len(pixel) != 3 for pixel in self.pixels):
            raise ValueError("each pixel must contain exactly three RGB channels")
        if any(not math.isfinite(channel) or channel < 0.0 or channel > 1.0 for pixel in self.pixels for channel in pixel):
            raise ValueError("RGB channels must be finite and normalized to [0, 1]")

    def pixel(self, x: int, y: int) -> tuple[float, float, float]:
        return self.pixels[y * self.width + x]

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[Sequence[float]]]) -> "RGBImage":
        if not rows or not rows[0]:
            raise ValueError("image rows must not be empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all image rows must have equal width")
        pixels = tuple(tuple(float(channel) for channel in pixel) for row in rows for pixel in row)
        return cls(width=width, height=len(rows), pixels=pixels)  # type: ignore[arg-type]


class ImageDecoder(Protocol):
    def decode(self, path: Path) -> RGBImage:
        """Decode a local file without downloading external resources."""


class PPMDecoder:
    """Dependency-free P3/P6 decoder for tests and reproducible fixtures."""

    def decode(self, path: Path) -> RGBImage:
        data = path.read_bytes()
        magic, width, height, maximum, offset = _ppm_header(data)
        expected_channels = width * height * 3
        if maximum <= 0 or maximum > 255:
            raise ImageDecodeError("only 8-bit PPM files are supported")

        if magic == b"P6":
            raw = data[offset : offset + expected_channels]
            if len(raw) != expected_channels or data[offset + expected_channels :].strip():
                raise ImageDecodeError("PPM pixel payload has an unexpected size")
            values = raw
        else:
            tokens = _ascii_tokens(data[offset:])
            if len(tokens) != expected_channels:
                raise ImageDecodeError("PPM pixel payload has an unexpected size")
            try:
                parsed = tuple(int(token) for token in tokens)
            except (ValueError, OverflowError) as exc:
                raise ImageDecodeError("invalid PPM channel value") from exc
            if any(value < 0 or value > maximum for value in parsed):
                raise ImageDecodeError("PPM channel value exceeds declared maximum")
            values = bytes(parsed)

        scale = float(maximum)
        pixels = tuple((values[i] / scale, values[i + 1] / scale, values[i + 2] / scale) for i in range(0, len(values), 3))
        return RGBImage(width=width, height=height, pixels=pixels)


def _ppm_header(data: bytes) -> tuple[bytes, int, int, int, int]:
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and not chr(data[index]).isspace():
            index += 1
        if start == index:
            raise ImageDecodeError("incomplete PPM header")
        tokens.append(data[start:index])

    if tokens[0] not in {b"P3", b"P6"}:
        raise ImageDecodeError("expected a P3 or P6 PPM image")
    try:
        width, height, maximum = (int(token) for token in tokens[1:])
    except ValueError as exc:
        raise ImageDecodeError("invalid PPM dimensions") from exc
    if width < 2 or height < 2 or width * height > MAX_PIXELS:
        raise ImageDecodeError(f"PPM dimensions must contain 4 to {MAX_PIXELS} pixels")
    if index >= len(data) or not chr(data[index]).isspace():
        raise ImageDecodeError("PPM header must end with whitespace")
    if tokens[0] == b"P6":
        index += 2 if data[index : index + 2] == b"\r\n" else 1
    else:
        while index < len(data) and chr(data[index]).isspace():
            index += 1
    return tokens[0], width, height, maximum, index


def _ascii_tokens(data: bytes) -> list[bytes]:
    without_comments = b"\n".join(line.split(b"#", 1)[0] for line in data.splitlines())
    return without_comments.split()
