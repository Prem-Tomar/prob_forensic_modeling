# Real-image baseline results

The first neural benchmark trains `SpatialFrequencyDetector` from scratch on a deduplicated CIFAKE partition. No pretrained weights are loaded. Calibration and the operating threshold are selected on validation data; the test partition is used only after those choices are fixed.

## Data audit

The downloaded CIFAKE release contains 120,000 files. Exact hashing found 1,336 duplicate groups and removed one redundant file from each group, leaving 118,664 unique files. The official train/test folders contained 378 cross-split duplicate groups, so the benchmark replaces them with a deterministic hash-group split: 83,037 training, 17,751 validation, and 17,876 test samples. Difference-hash screening identified 18 non-identical collision groups that require manual near-duplicate review.

## Results

| Test slice | AUROC | Balanced accuracy | ECE | FPR at 90% recall |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.8878 | 0.8040 | 0.0163 | 0.3322 |
| JPEG quality 30 | 0.8653 | 0.7809 | 0.0138 | 0.3972 |
| Gaussian blur radius 1 | 0.7862 | 0.7002 | 0.0906 | 0.5453 |
| 50% downscale and restore | 0.7639 | 0.6804 | 0.0699 | 0.5958 |

Clean AUROC has a 95% grouped-bootstrap interval of 0.8831–0.8921 from 500 resamples. The 76,113-parameter model reached 0.8835 validation AUROC after eight CPU epochs. The clean threshold was fixed at 0.4504 from validation balanced accuracy.

Pairing 1,000 held-out SynthScars synthetic images with 1,000 CIFAKE real controls produced 0.8672 AUROC and 0.7960 balanced accuracy. This is a diagnostic result, not an unseen-generator deployment claim: SynthScars is synthetic-only, so the pairing introduces source, content, and resolution confounds.

## Interpretation

The experiment demonstrates that a small scratch-trained model learns useful forensic evidence and transfers partially to an unrelated synthetic-image source. It also fails the robustness gate: blur and resizing materially degrade ranking and calibration. The next image experiments must add controlled spatial-only and frequency-only ablations, stronger post-processing augmentation, semantically matched real controls, generator-specific holdouts, and latency/memory measurements.

## Reproduction

Install the versions in `requirements-neural.lock`, download the credited datasets locally, and run:

```bash
PYTHONPATH=src python -m forensic_model.neural_experiment \
  --cifake-root data/raw/cifake/DATASET \
  --synthscars-root data/raw/synthscars/SynthScars \
  --output reports/real-image-evaluation.json \
  --checkpoint artifacts/spatial-frequency-v1.pt \
  --epochs 8 --batch-size 256 --seed 20260818
```

The JSON report records the complete learning curve, calibration parameters, threshold, environment versions, data audit, dataset credit, checkpoint hash, confidence metrics, grouped uncertainty, and branch-contribution summary. Raw media and the checkpoint are deliberately not redistributed.
