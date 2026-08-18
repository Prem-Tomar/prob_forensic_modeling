"""Provenance evidence contracts and conservative late fusion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from forensic_model.decision import DetectionResult


class ProvenanceStatus(str, Enum):
    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ProvenanceEvidence:
    status: ProvenanceStatus
    media_sha256: str
    credential_media_sha256: str | None = None
    signer: str | None = None
    generator_asserted: bool | None = None
    details: tuple[str, ...] = ()


class ProvenanceVerifier(Protocol):
    """Adapter boundary for a credential and trust-list implementation."""

    def verify(self, media_path: Path) -> ProvenanceEvidence:
        """Verify signature, trust, assertions, and asset binding."""


@dataclass(frozen=True)
class ForensicResult:
    decision: str
    conflict: bool
    pixel: DetectionResult
    provenance: ProvenanceEvidence
    rationale: tuple[str, ...]


def check_provenance(media_path: Path, verifier: ProvenanceVerifier | None = None) -> ProvenanceEvidence:
    """Compute local asset identity and validate a verifier's claimed binding."""

    media_hash = _sha256(media_path)
    if verifier is None:
        return ProvenanceEvidence(
            status=ProvenanceStatus.ABSENT,
            media_sha256=media_hash,
            details=("no supported credential was supplied",),
        )

    evidence = verifier.verify(media_path)
    if evidence.media_sha256.lower() != media_hash:
        return ProvenanceEvidence(
            status=ProvenanceStatus.INVALID,
            media_sha256=media_hash,
            credential_media_sha256=evidence.credential_media_sha256,
            signer=evidence.signer,
            generator_asserted=evidence.generator_asserted,
            details=evidence.details + ("verifier result was bound to different media bytes",),
        )
    if evidence.status == ProvenanceStatus.VALID and evidence.credential_media_sha256 != media_hash:
        return ProvenanceEvidence(
            status=ProvenanceStatus.INVALID,
            media_sha256=media_hash,
            credential_media_sha256=evidence.credential_media_sha256,
            signer=evidence.signer,
            generator_asserted=evidence.generator_asserted,
            details=evidence.details + ("credential asset hash does not match media bytes",),
        )
    return evidence


def fuse_evidence(pixel: DetectionResult, provenance: ProvenanceEvidence) -> ForensicResult:
    """Fuse decisions without hiding either source or converting absence into a label."""

    rationale = [f"pixel model decision: {pixel.decision}", f"provenance status: {provenance.status.value}"]
    if provenance.status in {ProvenanceStatus.ABSENT, ProvenanceStatus.UNSUPPORTED, ProvenanceStatus.INDETERMINATE}:
        rationale.append("provenance does not change the pixel decision")
        return ForensicResult(pixel.decision, False, pixel, provenance, tuple(rationale))

    if provenance.status == ProvenanceStatus.INVALID:
        rationale.append("invalid provenance forces abstention but is not evidence of synthesis")
        return ForensicResult("abstain", pixel.decision != "abstain", pixel, provenance, tuple(rationale))

    asserted = "synthetic" if provenance.generator_asserted is True else "camera_or_human" if provenance.generator_asserted is False else None
    if asserted is None:
        rationale.append("valid credential has no authorship assertion; pixel decision is preserved")
        return ForensicResult(pixel.decision, False, pixel, provenance, tuple(rationale))
    if pixel.decision == "abstain":
        rationale.append(f"verified assertion supplies the decision: {asserted}")
        return ForensicResult(asserted, False, pixel, provenance, tuple(rationale))
    if pixel.decision != asserted:
        rationale.append("verified assertion conflicts with the statistical model; review is required")
        return ForensicResult("abstain", True, pixel, provenance, tuple(rationale))
    rationale.append("verified assertion agrees with the statistical model")
    return ForensicResult(asserted, False, pixel, provenance, tuple(rationale))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
