# Frozen image post-processing stress results

This experiment evaluates the frozen neural and Phase 1 feature artifacts on the identical 17,876-row CIFAKE test split. It does not retrain either model or refit either calibrator or threshold. Each transformed row remains paired with its clean row by audited content identity.

## Results

| Operation | Neural AUROC | Neural 95% CI | Feature AUROC | Neural minus feature | Neural mean absolute probability shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.8244 | 0.8186–0.8298 | 0.7804 | +0.0441 | 0.0000 |
| JPEG quality 30 | 0.8193 | 0.8131–0.8261 | 0.7808 | +0.0385 | 0.0196 |
| WebP quality 30 | 0.8215 | 0.8156–0.8263 | 0.7766 | +0.0449 | 0.0195 |
| 50% resize and restore | 0.8187 | 0.8130–0.8243 | 0.6867 | +0.1321 | 0.0700 |
| Center crop 80% and restore | 0.7954 | 0.7883–0.8005 | 0.7595 | +0.0359 | 0.0920 |
| Gaussian blur radius 1 | 0.8220 | 0.8168–0.8286 | 0.7012 | +0.1209 | 0.0476 |
| Sharpness factor 2 | 0.8190 | 0.8128–0.8250 | 0.7801 | +0.0390 | 0.0217 |
| Deterministic RGB noise 0.02 | 0.8242 | 0.8184–0.8290 | 0.7855 | +0.0388 | 0.0012 |
| Gamma exponent 0.8 | 0.8067 | 0.8007–0.8126 | 0.7850 | +0.0218 | 0.1004 |
| Color saturation 0.7 | 0.8090 | 0.8023–0.8151 | 0.7800 | +0.0290 | 0.0605 |
| Screenshot simulation | 0.8102 | 0.8050–0.8175 | 0.7756 | +0.0346 | 0.1584 |
| Metadata removal | 0.8244 | 0.8195–0.8304 | 0.7804 | +0.0441 | 0.0000 |

Every interval uses 200 deterministic identity-group bootstrap resamples. The neural model ranks better than the feature baseline on every operation. Crop-and-rescale is its worst ranking slice at 0.7954 AUROC, a 0.0291 drop from clean. Screenshot simulation causes the largest mean absolute probability shift even though its AUROC remains 0.8102; this shows why ranking alone is not enough to characterize post-processing sensitivity.

The metadata-removal operation is deliberately a pixel-preserving PNG round trip. Its zero paired score change confirms that these pixel-only detectors do not infer provenance from removed metadata. The result must not be interpreted as a provenance-verification test.

## Reproduction

With the locally trained ignored artifacts and credited CIFAKE data present, run:

```bash
PYTHONPATH=src python -m forensic_model.image_stress_experiment \
  --cifake-root data/raw/cifake/DATASET \
  --neural-checkpoint artifacts/spatial-frequency-v1.pt \
  --feature-artifact artifacts/feature-baseline-v1.json \
  --output reports/image-stress-evaluation.json \
  --batch-size 512 --bootstrap-resamples 200 \
  --cpu-threads 8 --seed 20260818
```

The report records exact artifact hashes, the shared split-membership digest, complete metrics for both models, grouped AUROC intervals, and paired probability shifts. Its SHA-256 is `3c492f7e934806fa3aeb3a20d040d82c3bc2f670ae8077843688edaa23603ae9`.

## Scope

This closes the implementation gap for the specified deterministic post-processing matrix, not the full robustness phase. CIFAKE contains one synthetic generator family, the transforms are reference implementations rather than an exhaustive platform matrix, and no semantically matched real controls or additional held-out generator families are yet present.
