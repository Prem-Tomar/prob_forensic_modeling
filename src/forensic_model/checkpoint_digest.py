"""Canonical semantic identities for safe neural checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def semantic_checkpoint_sha256(path: Path) -> str:
    """Hash checkpoint meaning instead of serialization-container bytes."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return semantic_checkpoint_payload_sha256(payload)


def semantic_checkpoint_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable digest over format, config, metadata, and tensor values."""

    checkpoint_format = payload.get("format")
    config = payload.get("config")
    metadata = payload.get("metadata", {})
    state_dict = payload.get("state_dict")
    if not isinstance(checkpoint_format, str):
        raise ValueError("checkpoint format must be a string")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")

    digest = hashlib.sha256()
    _update_frame(
        digest,
        b"checkpoint",
        _canonical_json(
            {
                "format": checkpoint_format,
                "config": dict(config),
                "metadata": dict(metadata),
            }
        ),
    )
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("checkpoint state_dict must map string names to tensors")
        value = tensor.detach().cpu().contiguous()
        _update_frame(
            digest,
            b"tensor",
            _canonical_json(
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            ),
        )
        _update_frame(digest, b"values", value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must contain canonical JSON values") from error
    return encoded.encode("utf-8")


def _update_frame(digest: Any, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
