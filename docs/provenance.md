# Provenance evidence

Provenance and pixel forensics answer different questions. A cryptographically verified credential can bind assertions to exact media bytes and a trust policy; a pixel model estimates statistical similarity. Neither channel is silently converted into the other.

The library exposes a `ProvenanceVerifier` protocol for C2PA/content-credential adapters. A production adapter must validate the credential signature, certificate/trust-list policy, assertion structure, revocation state, and asset hash. `check_provenance` independently computes the media SHA-256 and rejects verifier results or credentials bound to different bytes.

Statuses are explicit: `absent`, `valid`, `invalid`, `unsupported`, and `indeterminate`. Missing or unsupported provenance leaves the pixel decision unchanged. Invalid provenance forces abstention but does not imply that the media is synthetic. A conflict between a valid authorship assertion and the pixel model also abstains and preserves both evidence records for review.

The repository does not include trust anchors, signing keys, third-party credentials, or a claim that the dependency-free protocol implementation itself validates C2PA signatures. Those belong in a separately licensed adapter and deployment trust policy.
