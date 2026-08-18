# Evaluation contract

## Data partitions

Each manifest row must include a stable sample ID, content-identity group, class label, source, source type, license identifier, generator family/version when applicable, capture device when applicable, semantic category, transformation lineage, file hash, and split.

Required partitions:

- **train**: model fitting;
- **validation**: architecture and hyperparameter selection;
- **calibration**: probability mapping, threshold choice, and abstention tuning;
- **in-distribution test**: frozen final comparison;
- **generator holdout**: entire synthetic generator families absent from earlier partitions;
- **source/device holdout**: real-media sources or devices absent from earlier partitions;
- **post-processing matrix**: paired originals and deterministic transformations;
- **video holdout**: clip-identity and generator-family grouped.

Content identities and derivatives may occur in exactly one partition. Near-duplicate detection is required before a real-data report is accepted.

## Metrics

The primary ranking metric is AUROC on the generator-family holdout. Because deployment prevalence varies, also report AUPRC with class counts and assumed prevalence.

Required decision and calibration metrics:

- balanced accuracy, sensitivity, specificity, precision, recall, and F1 at the frozen threshold;
- false-positive rate at 90% synthetic recall and synthetic recall at 1% false-positive rate;
- Brier score, log loss, expected calibration error, and reliability bins;
- coverage and selective risk for the abstaining system;
- bootstrap 95% confidence intervals grouped by content identity or clip;
- per-slice sample counts and Wilson intervals for rates.

Video adds clip-level metrics, time-to-decision, frame/clip disagreement, and processing latency per minute.

## Initial acceptance gates

These are engineering gates, not universal accuracy guarantees. They apply only to an approved benchmark with at least 1,000 content groups per class and at least three held-out generator families.

| Gate | Target |
| --- | ---: |
| Generator-holdout AUROC | >= 0.85 |
| Generator-holdout balanced accuracy | >= 0.75 |
| Generator-holdout Brier score | <= 0.18 |
| Expected calibration error (15 equal-mass bins) | <= 0.05 |
| Worst required post-processing AUROC drop | <= 0.10 absolute |
| Recall at 1% false-positive rate | reported, no hidden threshold retuning |
| Explanation/provenance schema validity | 100% |
| Deterministic rerun metric drift | 0 at fixed environment and seed |

The synthetic smoke dataset is exempt from accuracy gates and can validate only mechanics.

## Unseen-generator protocol

For each eligible generator family, train and tune without that family, freeze the model, calibrator, and threshold, then evaluate the held-out family. Aggregate macro averages across families and report every family independently. Do not select checkpoints using any held-out-family result.

## Post-processing protocol

Apply deterministic parameter grids to paired test images. At minimum include JPEG/WebP recompression, resize, crop-and-rescale, blur, sharpen, additive noise, gamma/color changes, screenshot simulation, and metadata removal. Report clean metrics, each operation/severity, worst case, and paired score change.

## Reproducibility record

Each report contains code revision, environment-lock hash, seed, manifest and file hashes, feature/model versions, configuration, command line, host capability summary, start/end timestamps, and all metrics including failed or unsupported slices.

## Licensing gate

Real training cannot begin until each dataset and pretrained weight has an explicit license/terms record permitting the intended research and distribution behavior. Unknown or incompatible assets are rejected by manifest validation. Raw media and third-party model weights are not committed.

## Reporting discipline

Do not claim real-world detection quality from synthetic fixtures, random image splits, a single generator, accuracy alone, or an uncalibrated score. Every comparison uses identical partitions and includes uncertainty and failures.
