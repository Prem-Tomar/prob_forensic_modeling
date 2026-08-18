# Probabilistic Forensic Modeling

This project studies whether learned forensic models can distinguish camera-originated media from generated or heavily synthesized media. The emphasis is on reproducible experiments, calibrated uncertainty, provenance as a separate evidence channel, and explicit testing against distribution shift.

The project starts with an auditable image baseline, then adds provenance evidence, confidence calibration and explanations, generator/post-processing holdouts, and temporal video analysis. See [ROADMAP.md](ROADMAP.md) for learning milestones and [docs/evaluation.md](docs/evaluation.md) for the evaluation contract.

## Current status

The repository is in Phase 0: architecture and evaluation design. No accuracy claim is valid until it is backed by a versioned evaluation report and an approved data manifest.

## Ground rules

- A detector score is evidence, not proof of authorship.
- Train, validation, calibration, and test identities must not overlap.
- Generator-family holdouts are required; random image splits are insufficient.
- Provenance and pixel-model evidence remain separate until the final fusion layer.
- Every published metric includes its dataset manifest, decision threshold, and uncertainty interval.
- Media, model weights, and secrets are never committed.

## Repository map

```text
docs/                 architecture, evaluation, and data governance
src/forensic_model/   reusable training and inference code
tests/                small deterministic tests
scripts/              reproducible local entry points
artifacts/             ignored local models and reports
data/                  ignored local datasets and manifests
```

## Development sequence

1. Establish the data and evaluation contract.
2. Train an interpretable image baseline and freeze its test protocol.
3. Add stronger spatial/frequency models without changing the protocol.
4. Add provenance, calibration, and per-prediction explanations.
5. Stress unseen generator families and post-processing transformations.
6. Extend inference to temporal video evidence.

The initial implementation intentionally avoids hidden downloads. Later model and dataset dependencies are introduced only after their licenses and hashes are recorded.
