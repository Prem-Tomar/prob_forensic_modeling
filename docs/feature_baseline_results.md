# Matched Phase 1 feature-baseline results

The interpretable Phase 1 model is standardized logistic regression over ten color, residual, edge, checkerboard, and clipping summaries. It was trained from scratch on the exact CIFAKE membership used by the neural model. Both reports record the same path-independent split digest: `2ae6738f68802a3d8c0a61a8195f8b378b1f9ac5c298f0d16550699431aee640`.

## Matched comparison

| Slice | Feature AUROC | Neural AUROC | Neural minus feature |
| --- | ---: | ---: | ---: |
| CIFAKE clean | 0.7804 | 0.8244 | +0.0441 |
| JPEG quality 30 | 0.7808 | 0.8193 | +0.0385 |
| Gaussian blur radius 1 | 0.7012 | 0.8220 | +0.1209 |
| 50% downscale and restore | 0.6867 | 0.8187 | +0.1321 |
| Diagnostic SynthScars/CIFAKE-real pairing | 0.9284 | 0.7232 | -0.2053 |

The feature model's clean grouped-bootstrap 95% AUROC interval is 0.7740–0.7863. On clean CIFAKE it reaches 0.7117 balanced accuracy, 0.1903 Brier score, and 0.0243 ECE. Blur and resizing expose severe fixed-threshold and calibration failure despite ranking remaining above chance: balanced accuracy falls near 0.50 and ECE rises above 0.32.

On the diagnostic SynthScars pairing, the feature model reaches 0.8230 balanced accuracy and 0.1243 Brier score, but its 0.1208 ECE still fails the calibration target. This pairing remains confounded by source, content, and resolution, and it represents only one synthetic holdout source.

## Learning result

The neural model clearly learns a stronger in-distribution and processing-robust representation. It does not satisfy the Phase 2 exit gate because its primary holdout ranking is substantially worse than the simpler model under the same split protocol. The likely lesson is shortcut replacement rather than generalization: removing the brittle frequency branch stabilized CIFAKE transformations, but the remaining spatial representation still specialized to CIFAKE content and source cues.

The next representation experiment must preserve the frozen split, add semantically matched controls and multiple generator families, and improve the macro generator-holdout result without losing the neural model's processing robustness.

## Reproduction

```bash
PYTHONPATH=src python -m forensic_model.feature_baseline_experiment \
  --cifake-root data/raw/cifake/DATASET \
  --synthscars-root data/raw/synthscars \
  --output reports/feature-baseline-evaluation.json \
  --artifact artifacts/feature-baseline-v1.json \
  --neural-report reports/real-image-evaluation.json \
  --epochs 800 --batch-size 512 --cpu-threads 8 --seed 20260818
```

The feature report SHA-256 is `5abb4e90c9a32a8f881e7d67f97a4f6e9e00f3218d2989f2b4554384a35b9658`; the ignored model artifact SHA-256 is `ae61add3c91fe085f2a460c976e1f5a0ec2fe510260dd941c3e519ae880466e6`.
