# Reproducible evaluation

Install the package locally, then run:

```bash
forensic-model smoke-evaluate --output reports/smoke-evaluation.json
```

The smoke experiment trains the image baseline, fits held-out calibration, evaluates in-distribution and two declared unseen procedural families, runs the deterministic processing matrix, computes identity-group bootstrap uncertainty, trains/calibrates the temporal model, and compares it with mean frame aggregation. It uses no downloads or third-party media.

The JSON is byte-stable for the same source tree and records a SHA-256 over the package implementation. Its `claim_scope` is `pipeline_mechanics_only`: procedural patterns can expose wiring errors but cannot establish useful real-world accuracy or satisfy acceptance gates.

## Offline wheel audit

Use a dedicated build environment populated from `requirements-build.lock`. `forensic_model.release` copies only package inputs into a temporary directory and invokes the locked build frontend with isolation disabled, so the build performs no dependency resolution or hidden download:

```bash
PYTHONPATH=src python -m forensic_model.release \
  --source-root . \
  --output-directory artifacts/release \
  --manifest reports/release-build.json
```

The builder fixes `SOURCE_DATE_EPOCH` to the ZIP-compatible 1980-01-01 epoch. Two consecutive builds from the same source produced the same wheel SHA-256, recorded in `reports/release-build.json`.

`forensic_model.release_verify` creates a fresh environment, installs the resulting wheel with `pip install --no-index --disable-pip-version-check --no-deps`, runs the installed smoke evaluator, and compares the output byte-for-byte with `reports/smoke-evaluation.json`:

```bash
PYTHONPATH=src python -m forensic_model.release_verify \
  --wheel artifacts/release/prob_forensic_modeling-0.1.0-py3-none-any.whl \
  --expected-smoke-report reports/smoke-evaluation.json \
  --output reports/release-verification.json
```

The recorded audit completed this core-library check successfully. It does not prove clean neural reproduction: that requires the pinned neural wheels, approved local datasets, ignored checkpoints, and sufficient CPU/storage to reproduce every real report without network access.

## Neural evidence reproduction

Torch checkpoint containers are not byte-canonical: two saves can differ even when their configuration, metadata, tensor names, dtypes, shapes, and values are identical. `checkpoint_semantic_sha256` hashes those semantic elements with length-delimited canonical encoding while retaining the raw file hash for transport-integrity checks.

Real reports declare a small, explicit list of volatile JSON fields. Compare a fresh run against the committed evidence with:

```bash
forensic-verify-report \
  --expected reports/real-image-evaluation.json \
  --candidate /tmp/real-image-reproduction.json \
  --output /tmp/real-image-reproduction-verification.json
```

The verifier requires both reports to declare the same sorted `semantic-report-v1` contract. It removes only those declared fields, hashes all remaining evidence as canonical JSON, writes the result before failing on a mismatch, and never treats timing or serialization-container differences as metric differences.

An exact-seed local rerun reproduced every image and video metric, model tensor, and checkpoint metadata. Raw checkpoint bytes and video timings differed as expected. This verifies deterministic training in the existing locked environment, but clean offline installation remains open until the pinned neural wheels are available locally.

For a real report, replace fixture construction with approved manifest-backed adapters while retaining the same split checks and evaluator. Record the environment lock, exact manifest/file hashes, model/weight license records, code revision, configuration, timing, hardware, all requested slices, uncertainty intervals, and failures. Never tune the model, calibrator, threshold, or abstention policy against final test results.

The frozen image stress report is reproduced independently of training:

```bash
PYTHONPATH=src python -m forensic_model.image_stress_experiment \
  --cifake-root data/raw/cifake/DATASET \
  --neural-checkpoint artifacts/spatial-frequency-v1.pt \
  --feature-artifact artifacts/feature-baseline-v1.json \
  --output reports/image-stress-evaluation.json \
  --batch-size 512 --bootstrap-resamples 200 \
  --cpu-threads 8 --seed 20260818
```

The evaluator cryptographically identifies both frozen artifacts and the split membership, uses fixed calibration and decision thresholds, and keeps transformed predictions paired with clean rows. The public report contains aggregate evidence only; datasets and model artifacts remain ignored.
