# Real-image baseline results

The first neural benchmark trains `SpatialFrequencyDetector` from scratch on a deduplicated CIFAKE partition. No pretrained weights are loaded. A validation-only robustness ablation selects the branch blend before calibration and threshold selection; the test partition is used only after those choices are fixed.

## Data audit

The downloaded CIFAKE release contains 120,000 files. Exact hashing found 1,336 duplicate groups and removed one redundant file from each group, leaving 118,664 unique files. The official train/test folders contained 378 cross-split duplicate groups, so the benchmark replaces them with a deterministic hash-group split: 83,037 training, 17,751 validation, and 17,876 test samples. Difference-hash screening identified 18 non-identical collision groups that require manual near-duplicate review.

## Results

| Test slice | AUROC | Balanced accuracy | ECE | FPR at 90% recall |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.8244 | 0.7447 | 0.0248 | 0.4896 |
| JPEG quality 30 | 0.8193 | 0.7406 | 0.0219 | 0.4978 |
| Gaussian blur radius 1 | 0.8220 | 0.7370 | 0.0463 | 0.4914 |
| 50% downscale and restore | 0.8187 | 0.7270 | 0.0656 | 0.4965 |

Clean AUROC has a 95% grouped-bootstrap interval of 0.8188–0.8300 from 500 resamples. The 76,113-parameter model reached 0.8835 full-model validation AUROC after eight CPU epochs. Across validation clean, JPEG, blur, and resize slices, the robust selector chose a frequency-branch weight of 0.0; calibration and the 0.4825 threshold were then refit on clean validation scores from that selected spatial-only blend.

Pairing 1,000 held-out SynthScars synthetic images with 1,000 CIFAKE real controls produced 0.7232 AUROC and 0.6595 balanced accuracy. This is a diagnostic result, not an unseen-generator deployment claim: SynthScars is synthetic-only, so the pairing introduces source, content, and resolution confounds.

The matched 11-parameter Phase 1 feature model reaches only 0.7804 clean CIFAKE AUROC but 0.9284 on the same diagnostic SynthScars pairing. The reports share split membership digest `2ae6738f68802a3d8c0a61a8195f8b378b1f9ac5c298f0d16550699431aee640`; see [feature_baseline_results.md](feature_baseline_results.md). The neural representation therefore improves every CIFAKE processing slice but fails the primary holdout comparison.

## Interpretation

The validation-selected spatial-only blend reduced clean AUROC by 0.0634 compared with the earlier full-branch run, but reduced the worst required stress drop from 0.1239 to 0.0057. This is the intended lesson from the controlled ablation: the learned frequency branch improved in-distribution ranking while making the detector brittle to blur and resizing. The checkpoint keeps both learned branches and records the selected weight, so explanations expose the raw branch evidence while clearly reporting that the deployed frequency contribution is zero.

The original four-slice post-processing-drop gate is met. The frozen-artifact follow-up expands this to WebP, crop-and-rescale, sharpen, deterministic noise, gamma, color, screenshot simulation, and metadata removal with paired probability shifts and grouped uncertainty; see [image_stress_results.md](image_stress_results.md). The overall image phase remains open because the diagnostic unseen-source result misses the initial ranking, accuracy, Brier, and calibration targets, and the approved protocol still requires at least three generator-family holdouts and semantically matched real controls.

## Compute measurement

The calibrated public inference path was measured on the report's ARM64 CPU environment after five warmup iterations. A 64-image in-memory tensor batch averaged 0.369 ms per image (2,708 images/second), with a 23.64 ms median batch latency. Local JPEG open, RGB conversion, resize, inference, calibration, explanation, and decision took a 0.798 ms median per file and 0.940 ms at the 95th percentile over 64 warm-cache iterations.

The learned parameters occupy 304,452 bytes; the checkpoint occupies 310,717 bytes; and the 64-image float32 input batch occupies 786,432 bytes. Peak process RSS was 343,523,328 bytes, which includes the Python and PyTorch runtimes and is reported as a host observation rather than a model-only allocation.

## Reproduction

Install the versions in `requirements-neural.lock`, download the credited datasets locally, and run:

```bash
PYTHONPATH=src python -m forensic_model.neural_experiment \
  --cifake-root data/raw/cifake/DATASET \
  --synthscars-root data/raw/synthscars \
  --output reports/real-image-evaluation.json \
  --checkpoint artifacts/spatial-frequency-v1.pt \
  --epochs 8 --batch-size 256 --seed 20260818
```

The JSON report records the complete learning curve, calibration parameters, threshold, environment versions, data audit, dataset credit, checkpoint hash, confidence metrics, grouped uncertainty, and branch-contribution summary. Raw media and the checkpoint are deliberately not redistributed.

The report SHA-256 is `1988181ffe231fa2f63de03d5df57490e25af78a9900518c3f5e9343fe9101dd`. The ignored checkpoint SHA-256 is `fd9b220f746a5e06c6c5b866797b605acd6cd10d5778fb66fec5bacba101642a`.

Reproduce the compute measurement with:

```bash
PYTHONPATH=src python -m forensic_model.neural_benchmark \
  --checkpoint artifacts/spatial-frequency-v1.pt \
  --image-root data/raw/cifake/DATASET \
  --output reports/image-compute-evaluation.json \
  --sample-count 32 --image-size 32 --batch-size 64 \
  --warmup-iterations 5 --tensor-iterations 50 --file-iterations 64 \
  --cpu-threads 8 --seed 20260818
```

The compute report SHA-256 is `e0bc8ffe16de23c10476a82535584db84577004b5e5a1fe6574067a28e0027bf`.
