# Learning-first roadmap

Each phase is a learning milestone with an exit gate. A phase is complete only when its code, tests, documentation, and reproducible report satisfy the gate.

## Implementation status

| Phase | Implemented | Remaining exit gate |
| --- | --- | --- |
| 0 — Claim and data contract | Architecture, attributed manifest licensing, hash/path/split/leakage validation, metrics, reproducible CIFAKE collision review | Choose grouping or exclusion for one semantic pair and run the gate on all approved real manifests |
| 1 — Image MVP | Decoder boundary, interpretable feature model, exact split digest, matched deduplicated CIFAKE/SynthScars report, generic matched-family evaluator | Supply three eligible generator holdouts and meet their calibrated acceptance metrics |
| 2 — Stronger representations | Scratch spatial-frequency model, validation-only branch ablation, reusable inference API, matched Phase 1 comparison, latency and memory report | Improve the primary generator holdout without losing processing robustness |
| 3 — Provenance and confidence | Calibration, abstention, reliability bins, Wilson rate uncertainty, coverage/selective risk, contributions, safe local verifier adapter, signer trust policy, late fusion, attributed hash-bound conformance runner | Run the conformance gate with an approved C2PA verifier and licensed public fixtures |
| 4 — Robustness | Full deterministic CIFAKE codec/post-processing matrix, paired degradation, grouped uncertainty, diagnostic SynthScars transfer, governed source/device/category slice evaluator | Three or more generator holdouts with complete slice metadata, matched controls, and worst-slice acceptance |
| 5 — Video | Scene-aware and uniform sampling, PyAV adapter, audited DAVIS/Keling T2V+I2V/Sora holdouts, reusable calibrated temporal inference, portable matched-video manifests, frozen frame comparison, stress and latency reports | Supply eligible matched controls and another training generator, then beat frame aggregation |
| 6 — Release | Smoke/image/video reports, build and neural locks, deterministic wheel, semantic checkpoint/report verification, exact-seed local neural reproduction | Clean offline neural installation and final evidence-to-claim audit |

The real-image report establishes a research baseline, not production assurance. Rows remain open until every listed robustness, provenance, video, and release gate is satisfied.

## Phase 0 — Frame the forensic claim

Learn how leakage, generator memorization, prevalence, and threshold selection can make a detector look stronger than it is.

Deliverables:

- threat model and evidence boundaries;
- data manifest with source, license, generator family, transformation lineage, and content identity;
- immutable split policy grouped by content identity and generator family;
- metric definitions and acceptance gates.

Exit gate: a dry-run evaluator rejects overlapping identities, missing licenses, and test-time threshold tuning.

## Phase 1 — Image-detection MVP

Learn the full train-to-inference loop with a small, inspectable classifier before increasing model capacity.

Deliverables:

- deterministic image decoding and feature extraction;
- learned baseline over spatial, color, noise, and frequency-summary features;
- model card containing feature version, training-manifest hash, threshold, and metrics;
- CLI for training, evaluation, and single-image prediction;
- unit tests plus a synthetic smoke dataset that tests mechanics, not real-world accuracy.

Exit gate: identical inputs produce identical artifacts; test AUROC, AUPRC, balanced accuracy, Brier score, and expected calibration error are reported with bootstrap intervals. Real-world acceptance targets appear in `docs/evaluation.md`.

## Phase 2 — Stronger image representations

Learn which signals generalize rather than merely identifying known generators.

Deliverables:

- patch-based convolutional or vision-transformer backbone with documented weight license;
- spatial and frequency branches with controlled ablations;
- comparison with the Phase 1 baseline under the unchanged split protocol;
- compute, latency, memory, and failure-mode reporting.

Exit gate: the stronger model improves the primary holdout metric without worsening calibration or the worst transformation slice beyond the agreed tolerance.

## Phase 3 — Provenance and explainable confidence

Learn to distinguish cryptographic/content credentials from statistical inference.

Deliverables:

- provenance adapter reporting absent, valid, invalid, unsupported, or indeterminate evidence;
- score calibration fitted only on the calibration split;
- abstention policy for low-confidence and out-of-scope inputs;
- feature/region contributions and plain-language reason codes;
- late-fusion policy preserving both component results.

Exit gate: provenance failures never silently become a synthetic label, calibration improves held-out Brier/ECE, and explanations pass stability and sanity tests.

## Phase 4 — Robustness to unseen generators and processing

Learn how quickly forensic cues fail under realistic distribution shift.

Deliverables:

- leave-one-generator-family-out evaluation;
- post-processing matrix covering resize, crop, recompression, blur, sharpen, noise, color changes, screenshots, and metadata removal;
- source/device and semantic-category slices;
- paired degradation curves and bootstrap uncertainty;
- documented failure catalog.

Exit gate: all required slices are present, no holdout family influenced training or threshold selection, and worst-slice metrics meet the evaluation gates.

## Phase 5 — Temporal video analysis

Learn when temporal consistency adds evidence beyond averaging frame scores.

Deliverables:

- deterministic frame sampling with scene-aware coverage;
- image evidence plus temporal features for score dynamics, optical consistency, and spectral flicker;
- temporal sequence model and simple aggregation baseline;
- clip-level calibration, abstention, and timestamped explanations;
- codec, frame-rate, duration, and generator-family holdouts.

Exit gate: temporal modeling beats the matched frame-aggregation baseline on clip-level holdouts and reports latency per minute of video.

## Phase 6 — Reproducible release

Deliverables:

- locked environment and local reproduction commands;
- approved data/model license inventory and artifact hashes;
- versioned model cards and evaluation reports;
- completion audit mapping every claim to code, data, and report evidence.

Exit gate: a clean machine with approved local inputs can recreate the reported metrics without network access.

## Commit curriculum

Changes remain small and concept-focused: evaluation contract, data validation, baseline features, model learning, calibration, provenance, robustness, temporal sampling, temporal modeling, and reporting. Tests and concise documentation travel with the concept they teach.
