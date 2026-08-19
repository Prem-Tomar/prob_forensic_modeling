# Provenance evidence

Provenance and pixel forensics answer different questions. A cryptographically verified credential can bind assertions to exact media bytes and a trust policy; a pixel model estimates statistical similarity. Neither channel is silently converted into the other.

The library exposes a `ProvenanceVerifier` protocol for C2PA/content-credential adapters. A production adapter must validate the credential signature, certificate/trust-list policy, assertion structure, revocation state, and asset hash. `check_provenance` independently computes the media SHA-256 and rejects verifier results or credentials bound to different bytes.

Statuses are explicit: `absent`, `valid`, `invalid`, `unsupported`, and `indeterminate`. Missing or unsupported provenance leaves the pixel decision unchanged. Invalid provenance forces abstention but does not imply that the media is synthetic. A conflict between a valid authorship assertion and the pixel model also abstains and preserves both evidence records for review.

`JsonProcessVerifier` is a dependency-free adapter for an installed credential verifier. It never invokes a shell, never inherits the parent environment, suppresses verifier stderr, applies execution and output limits, parses a strict JSON schema, and then applies a deployment-owned `SignerTrustPolicy`. The media path is appended as the final process argument.

The verifier process must return one JSON object with `status`, `media_sha256`, optional `credential_media_sha256`, optional `signer`, optional boolean `generator_asserted`, and a string-list `details`. A `valid` result becomes `indeterminate` unless its signer is explicitly allowlisted or the deployment deliberately enables unlisted signers. `check_provenance` still recomputes the asset hash independently after the adapter returns.

The repository does not include trust anchors, signing keys, third-party credentials, or a claim that this process adapter itself validates C2PA signatures. A deployment must supply a separately licensed local verifier, its trust material, and revocation policy. The remaining conformance gate is an approved verifier tested against signed, tampered, revoked, unsupported, and absent-credential fixtures.
