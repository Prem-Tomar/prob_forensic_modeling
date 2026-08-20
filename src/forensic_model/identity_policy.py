"""Explicit reviewed resolutions for ambiguous near-duplicate identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from forensic_model.data_audit import CandidateSample


IDENTITY_POLICY_SCHEMA = "reviewed-collision-policy-v1"


@dataclass(frozen=True)
class IdentityPolicyAudit:
    schema: str
    policy_sha256: str
    reviewed_content_groups: int
    grouped_observations: int
    excluded_observations: int


def apply_identity_policy(
    candidates: Sequence[CandidateSample],
    policy_path: Path,
) -> tuple[tuple[CandidateSample, ...], IdentityPolicyAudit]:
    """Apply only explicit grouping and exclusion decisions by exact hash."""

    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read reviewed identity policy") from error
    if not isinstance(policy, dict) or policy.get("schema") != IDENTITY_POLICY_SCHEMA:
        raise ValueError(f"identity policy must use schema {IDENTITY_POLICY_SCHEMA}")
    raw_groups = policy.get("group_sha256", [])
    raw_exclusions = policy.get("exclude_sha256", [])
    if not isinstance(raw_groups, list) or not isinstance(raw_exclusions, list):
        raise ValueError("identity policy groups and exclusions must be lists")

    by_hash: dict[str, list[CandidateSample]] = {}
    for candidate in candidates:
        by_hash.setdefault(candidate.sha256.lower(), []).append(candidate)
    grouped: dict[str, str] = {}
    decisions: set[str] = set()
    grouped_observations = 0
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or len(raw_group) < 2:
            raise ValueError("each reviewed identity group must contain at least two hashes")
        hashes = tuple(sorted(_validated_hash(value) for value in raw_group))
        if len(set(hashes)) != len(hashes):
            raise ValueError("reviewed identity group contains duplicate hashes")
        _claim_decisions(hashes, decisions)
        members = [candidate for digest in hashes for candidate in by_hash.get(digest, ())]
        if len({candidate.label for candidate in members}) != 1:
            raise ValueError("reviewed identity group must contain one label")
        if any(digest not in by_hash for digest in hashes):
            raise ValueError("reviewed identity policy references an unknown hash")
        content_group = "reviewed-" + hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
        grouped.update({digest: content_group for digest in hashes})
        grouped_observations += len(members)
    exclusions = tuple(sorted(_validated_hash(value) for value in raw_exclusions))
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("identity policy contains duplicate exclusions")
    _claim_decisions(exclusions, decisions)
    if any(digest not in by_hash for digest in exclusions):
        raise ValueError("reviewed identity policy references an unknown hash")

    resolved = tuple(
        replace(candidate, reviewed_content_group=grouped.get(candidate.sha256.lower(), ""))
        for candidate in candidates
        if candidate.sha256.lower() not in exclusions
    )
    return resolved, IdentityPolicyAudit(
        IDENTITY_POLICY_SCHEMA,
        _sha256(policy_path),
        len(raw_groups),
        grouped_observations,
        sum(len(by_hash[digest]) for digest in exclusions),
    )


def _validated_hash(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("identity policy hashes must be strings")
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("identity policy contains an invalid SHA-256")
    return digest


def _claim_decisions(hashes: Sequence[str], decisions: set[str]) -> None:
    overlap = decisions.intersection(hashes)
    if overlap:
        raise ValueError("each identity hash may have only one reviewed decision")
    decisions.update(hashes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
