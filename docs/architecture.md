# Architecture and threat model

## Question being answered

Given media bytes and optional provenance records, estimate evidence for two operational classes:

- `camera_or_human`: camera-originated or conventionally edited media;
- `synthetic`: media substantially rendered by a generative system.

The label is a dataset policy, not an assertion about intent or ownership. Mixed and unknown cases should be represented explicitly and may be routed to abstention.

## Threat model

The system must expect unseen generator families, loss of metadata, recompression, resizing, screenshots, adversarially selected examples, content/source imbalance, and manipulated provenance. It does not claim to identify a specific generator, prove authenticity, detect every edit, or replace source verification.

## Evidence flow

```text
media bytes
  ├─ safety and format validation
  ├─ pixel model -> raw score -> calibrated probability ─┐
  ├─ out-of-scope checks ────────────────────────────────┤
  └─ provenance parser -> structured evidence ──────────┤
                                                          ├─> fusion and abstention
context policy ───────────────────────────────────────────┘          │
                                                                    └─> decision, uncertainty, reasons
```

## Components

1. **Input boundary** validates size, format, decoding limits, and unsupported media without interpreting content.
2. **Pixel model** begins as a learned linear baseline over versioned forensic features, then gains patch-level neural spatial and frequency branches.
3. **Calibration layer** maps raw scores to probabilities using only a held-out calibration split.
4. **Provenance adapter** validates supported credentials and reports structured status. Absence is never treated as evidence of synthesis.
5. **Fusion policy** combines independently visible evidence channels and may abstain when evidence conflicts or the input is out of scope.
6. **Explanation layer** exposes model version, calibrated confidence, decision threshold, uncertainty band, provenance status, and bounded reason codes.
7. **Evaluator** owns split validation, slice metrics, uncertainty estimates, artifact hashes, and comparison reports.
8. **Video pipeline** samples frames deterministically, reuses image evidence, extracts temporal residuals, and compares a learned sequence model with a frame-aggregation baseline.

## Dependency boundaries

Core schemas, split validation, metrics, and the initial baseline stay lightweight and deterministic. Image/video codecs, neural training, and provenance libraries are adapters behind narrow interfaces. No component downloads weights or data during training or inference.

## Artifacts

Every model artifact records:

- semantic model and feature versions;
- code revision and configuration hash;
- training and calibration manifest hashes;
- label policy and approved licenses;
- learned parameters and threshold;
- metric summary, evaluated slices, and known limitations.

Large inputs and artifacts live outside Git. Only small synthetic fixtures and machine-readable reports may be versioned.

## Key architectural decisions

- Use late fusion so provenance and statistical evidence remain auditable.
- Group splits by content identity; hold out entire generator families for generalization studies.
- Fit thresholds and calibrators without touching the final test set.
- Require an abstention path rather than forcing every input into a binary claim.
- Compare each complex model with a simple baseline under identical data and metrics.
