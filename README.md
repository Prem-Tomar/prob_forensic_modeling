# Probabilistic Forensic Modeling

This project studies whether learned forensic models can distinguish camera-originated media from generated or heavily synthesized media. The emphasis is on reproducible experiments, calibrated uncertainty, provenance as a separate evidence channel, and explicit testing against distribution shift.

The project starts with an auditable image baseline, then adds provenance evidence, confidence calibration and explanations, generator/post-processing holdouts, and temporal video analysis. See [ROADMAP.md](ROADMAP.md) for learning milestones and [docs/evaluation.md](docs/evaluation.md) for the evaluation contract.

## Current status

The library now includes dependency-free and neural image/video detectors trained from scratch. The neural image model improves matched CIFAKE clean AUROC from 0.7804 to 0.8244 and remains at or above 0.7954 across a 12-operation post-processing matrix, but trails the simple feature model on the diagnostic SynthScars pairing, 0.7232 versus 0.9284. On the Keling-to-Sora video holdout, adding Keling I2V mode diversity improves temporal AUROC from 0.5752 to 0.6273, while frozen frame aggregation remains stronger at 0.7442. Both stronger-model adoption gates therefore remain open; see [docs/feature_baseline_results.md](docs/feature_baseline_results.md), [docs/image_stress_results.md](docs/image_stress_results.md), [docs/real_image_results.md](docs/real_image_results.md), and [docs/real_video_results.md](docs/real_video_results.md). These are research baselines, not production assurance.

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
artifacts/             ignored local models and reports
data/                  ignored local datasets and manifests
requirements-*.lock   pinned build and neural environments
```

## Development sequence

1. Establish the data and evaluation contract.
2. Train an interpretable image baseline and freeze its test protocol.
3. Add stronger spatial/frequency models without changing the protocol.
4. Add provenance, calibration, and per-prediction explanations.
5. Stress unseen generator families and post-processing transformations.
6. Extend inference to temporal video evidence.

The initial implementation intentionally avoids hidden downloads. Later model and dataset dependencies are introduced only after their licenses and hashes are recorded.

## Library design

`forensic_model` is an installable Python library. Stable public functions and typed result objects own validation, training, prediction, explanation, and evaluation; command-line entry points only translate files and arguments into those APIs. Applications can therefore embed the detector without parsing terminal output or depending on project scripts.

## Reproduce the smoke evaluation

```bash
PYTHONPATH=src python3 -m forensic_model.cli smoke-evaluate --output reports/smoke-evaluation.json
```

This deterministic report validates the end-to-end experiment plumbing with procedural fixtures only. It is not evidence of real-world detector accuracy.

## Reproduce the real-image baseline

Install `requirements-neural.lock`, place the credited CIFAKE and SynthScars datasets under ignored local data directories, then follow [docs/neural_training.md](docs/neural_training.md). `forensic-train-feature-baseline` and `forensic-train-image` train the matched Phase 1 and neural models from scratch and write aggregate reports plus ignored local artifacts.

Applications with the neural extra installed can load the calibrated checkpoint directly:

```python
from pathlib import Path
from forensic_model.neural_inference import CalibratedNeuralImageDetector

detector = CalibratedNeuralImageDetector.load(Path("artifacts/spatial-frequency-v1.pt"))
result = detector.predict_file(Path("candidate.jpg"))
print(result.decision, result.probability_synthetic)
print(result.spatial_contribution, result.deployed_frequency_contribution)
```

`forensic-benchmark-image` measures this same public inference path, including both in-memory batch throughput and local file decode-to-decision latency.

`forensic-evaluate-image-stress` evaluates the frozen neural and Phase 1 artifacts on identical clean and transformed rows. It reports complete metrics, identity-group bootstrap intervals, and paired probability shifts without retraining or retuning either model.

`forensic-review-image-collisions` creates a path-free aggregate inventory of perceptual-hash collisions and an optional ignored contact sheet. Compact-hash collisions are measured and reviewed; they are never silently merged.

## Reproduce the real-video baseline

Install the same lock file, add the credited DAVIS 2017 and GenVidBench subsets under ignored local data directories, then follow [docs/real_video_results.md](docs/real_video_results.md). The `forensic-train-video` entry point performs source-grouped training, validation-only calibration, unseen-generator evaluation, and post-processing stress tests.

## Build an offline-installable library wheel

Install `requirements-build.lock` in a dedicated local build environment, then run:

```bash
PYTHONPATH=src python -m forensic_model.release \
  --source-root . \
  --output-directory artifacts/release \
  --manifest reports/release-build.json
```

The builder stages only `pyproject.toml`, `README.md`, and `src/` in a temporary directory, disables build isolation to prevent hidden dependency downloads, and records the wheel and lock-file hashes. See [docs/reproducibility.md](docs/reproducibility.md) and [docs/completion_audit.md](docs/completion_audit.md) for the verified scope and remaining gates.

Verify the wheel in a newly created dependency-free environment:

```bash
PYTHONPATH=src python -m forensic_model.release_verify \
  --wheel artifacts/release/prob_forensic_modeling-0.1.0-py3-none-any.whl \
  --expected-smoke-report reports/smoke-evaluation.json \
  --output reports/release-verification.json
```
