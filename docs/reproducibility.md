# Reproducible evaluation

Install the package locally, then run:

```bash
forensic-model smoke-evaluate --output reports/smoke-evaluation.json
```

The smoke experiment trains the image baseline, fits held-out calibration, evaluates in-distribution and two declared unseen procedural families, runs the deterministic processing matrix, computes identity-group bootstrap uncertainty, trains/calibrates the temporal model, and compares it with mean frame aggregation. It uses no downloads or third-party media.

The JSON is byte-stable for the same source tree and records a SHA-256 over the package implementation. Its `claim_scope` is `pipeline_mechanics_only`: procedural patterns can expose wiring errors but cannot establish useful real-world accuracy or satisfy acceptance gates.

For a real report, replace fixture construction with approved manifest-backed adapters while retaining the same split checks and evaluator. Record the environment lock, exact manifest/file hashes, model/weight license records, code revision, configuration, timing, hardware, all requested slices, uncertainty intervals, and failures. Never tune the model, calibrator, threshold, or abstention policy against final test results.
