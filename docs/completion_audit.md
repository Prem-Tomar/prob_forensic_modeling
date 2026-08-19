# Roadmap completion audit

This audit treats a phase as complete only when its exit gate is supported by executable code, tests, and a reproducible real-data report. A passing mechanics test is not substituted for distribution-shift evidence.

| Phase | Evidence present | Evidence still required | Status |
| --- | --- | --- | --- |
| 0 — Claim and data contract | Threat model, licensed manifest validation, identity-group splitting, exact duplicate audit, reproducible CIFAKE collision inventory and visual review, grouped metrics | Select grouping versus canonical exclusion for the one reviewed semantic pair, then demonstrate the same near-duplicate gate on every approved real manifest | Open |
| 1 — Image MVP | Deterministic decoder boundary, interpretable feature model, calibration, explanations, matched CIFAKE/SynthScars report, exact split digest | Meet the approved unseen-generator ranking, calibration, and decision gates on at least three held-out families | Open |
| 2 — Stronger representations | Scratch spatial/frequency model, validation-only branch selection, calibrated checkpoint API, latency/memory report, cryptographically matched Phase 1 comparison | Improve the primary generator holdout without losing processing robustness; the current neural model loses 0.2053 AUROC on the diagnostic holdout | Open |
| 3 — Provenance and confidence | Explicit provenance states, byte binding, late fusion, abstention, safe local process adapter, signer trust policy | Run an approved C2PA verifier against signed, tampered, revoked, unsupported, and absent-credential conformance fixtures | Open |
| 4 — Robustness | Full deterministic CIFAKE codec/post-processing matrix, paired probability shifts, grouped uncertainty, diagnostic SynthScars transfer | Add semantically matched controls and three or more generator holdouts; meet the worst-slice gates without holdout tuning | Open |
| 5 — Temporal video | Deterministic sampling/decoding, DAVIS/Keling T2V+I2V/Sora source audit, scratch temporal model, matched frame baseline, clip calibration, timing | Add matched real/generated controls and another training generator family; temporal AUROC must exceed the frozen frame baseline | Open |
| 6 — Reproducible release | Locked neural/build environments, hashed reports, byte-identical consecutive wheel builds, byte-identical smoke report from a fresh no-index wheel-only environment | Reproduce the neural image/video reports from approved local inputs in a clean offline environment and publish the final evidence-to-claim audit | Open |

## Current acceptance evidence

- Image clean AUROC: 0.8244; worst of the 12 implemented processing slices: 0.7954 on crop-and-rescale. Every slice includes a grouped 95% AUROC interval and paired probability-shift summary.
- Diagnostic SynthScars/CIFAKE-control AUROC: 0.9284 for the feature model and 0.7232 for the neural model; only the feature model exceeds the ranking target, and its ECE remains 0.1208.
- Frozen frame-aggregation AUROC: 0.7442; mode-diverse temporal AUROC: 0.6273. Adding Keling I2V improves temporal ranking by 0.0521, but the adoption gate still fails by 0.1169 AUROC.
- The local wheel installs without dependencies into a fresh Python environment, and its smoke report is byte-identical to the source-tree smoke report.

The remaining items are functional and evidence gaps, not documentation-only tasks. Final completion requires closing each row rather than relabeling the existing research baselines as production assurance.
