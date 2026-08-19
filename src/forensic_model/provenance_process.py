"""Safe local-process adapter for credential verification implementations."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from forensic_model.provenance import ProvenanceEvidence, ProvenanceStatus


@dataclass(frozen=True)
class SignerTrustPolicy:
    """Deployment-owned signer allowlist applied after cryptographic verification."""

    trusted_signers: frozenset[str] = frozenset()
    allow_unlisted_signers: bool = False

    def apply(self, evidence: ProvenanceEvidence) -> ProvenanceEvidence:
        if evidence.status != ProvenanceStatus.VALID:
            return evidence
        if evidence.signer is None:
            return _indeterminate(evidence, "valid credential did not identify a signer")
        if evidence.signer not in self.trusted_signers and not self.allow_unlisted_signers:
            return _indeterminate(evidence, "credential signer is not trusted by deployment policy")
        return evidence


@dataclass(frozen=True)
class JsonProcessVerifier:
    """Invoke an installed verifier without a shell and parse a strict JSON result."""

    command: tuple[str, ...]
    trust_policy: SignerTrustPolicy = SignerTrustPolicy()
    timeout_seconds: float = 10.0
    max_output_bytes: int = 1024 * 1024
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.command or any(not argument for argument in self.command):
            raise ValueError("verifier command must contain non-empty arguments")
        if self.timeout_seconds <= 0.0 or self.max_output_bytes <= 0:
            raise ValueError("verifier limits must be positive")

    def verify(self, media_path: Path) -> ProvenanceEvidence:
        media_hash = _sha256(media_path)
        try:
            result = subprocess.run(
                (*self.command, str(media_path)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self.timeout_seconds,
                env=dict(self.environment),
            )
        except FileNotFoundError:
            return ProvenanceEvidence(
                ProvenanceStatus.UNSUPPORTED,
                media_hash,
                details=("configured credential verifier is not installed",),
            )
        except subprocess.TimeoutExpired:
            return ProvenanceEvidence(
                ProvenanceStatus.INDETERMINATE,
                media_hash,
                details=("credential verifier exceeded its time limit",),
            )
        if result.returncode != 0:
            return ProvenanceEvidence(
                ProvenanceStatus.INDETERMINATE,
                media_hash,
                details=(f"credential verifier exited with status {result.returncode}",),
            )
        if len(result.stdout) > self.max_output_bytes:
            return ProvenanceEvidence(
                ProvenanceStatus.INDETERMINATE,
                media_hash,
                details=("credential verifier output exceeded its size limit",),
            )
        try:
            values = json.loads(result.stdout)
            evidence = _parse_evidence(values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ProvenanceEvidence(
                ProvenanceStatus.INDETERMINATE,
                media_hash,
                details=("credential verifier returned an invalid result",),
            )
        return self.trust_policy.apply(evidence)


def _parse_evidence(values: object) -> ProvenanceEvidence:
    if not isinstance(values, Mapping):
        raise TypeError("verifier result must be an object")
    status = ProvenanceStatus(str(values["status"]))
    media_hash = _digest(values["media_sha256"])
    credential_hash = values.get("credential_media_sha256")
    if credential_hash is not None:
        credential_hash = _digest(credential_hash)
    signer = values.get("signer")
    if signer is not None and not isinstance(signer, str):
        raise TypeError("signer must be a string or null")
    asserted = values.get("generator_asserted")
    if asserted is not None and not isinstance(asserted, bool):
        raise TypeError("generator_asserted must be boolean or null")
    details = values.get("details", [])
    if not isinstance(details, list) or any(not isinstance(detail, str) for detail in details):
        raise TypeError("details must be a string list")
    return ProvenanceEvidence(status, media_hash, credential_hash, signer, asserted, tuple(details))


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("asset hash must be a SHA-256 hex digest")
    int(value, 16)
    return value.lower()


def _indeterminate(evidence: ProvenanceEvidence, detail: str) -> ProvenanceEvidence:
    return ProvenanceEvidence(
        ProvenanceStatus.INDETERMINATE,
        evidence.media_sha256,
        evidence.credential_media_sha256,
        evidence.signer,
        evidence.generator_asserted,
        evidence.details + (detail,),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
