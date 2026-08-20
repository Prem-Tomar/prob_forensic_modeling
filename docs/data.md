# Data governance

## Manifest format

Experiments use UTF-8 CSV manifests. Paths refer to local files that are not committed. Required columns are enforced by `forensic_model.manifest`:

| Column | Purpose |
| --- | --- |
| `sample_id` | Stable row identity |
| `content_group` | Identity shared by originals and every derivative |
| `path` | Local media path |
| `sha256` | Exact file identity |
| `label` | `camera_or_human` or `synthetic` |
| `split` | Frozen experimental partition |
| `source` | Dataset or collection name |
| `source_url` | Official HTTP(S) dataset or publication location |
| `source_type` | Camera, curated, procedural, generated, or other documented type |
| `license_id` | Reviewed license/terms identifier |
| `citation` | Human-readable publication or dataset credit |
| `generator_family` | Required for synthetic samples |
| `transformation` | Original or ordered processing lineage |
| `semantic_category` | Required content category used for robustness slices |
| `capture_device` | Required for camera rows; matched generated rows inherit their control device during evaluation |

## Admission checklist

Before training, record the source terms, allowed research/commercial use, redistribution limits, attribution requirements, privacy/consent conditions, model-weight license, collection date, and reviewer decision outside the media directory. Unknown licenses are rejected.

## Leakage controls

- A file hash and sample ID are unique.
- All derivatives of a content identity stay in one split.
- Generator-holdout families are absent from train, validation, and calibration.
- `source_holdout` accepts only real rows and requires each row to introduce a source or capture device absent from train, validation, and calibration.
- Raw collections are deduplicated before manifest creation with `partition_candidates`. Exact-byte groups are kept in one deterministic split, while perceptual-hash collisions are reported for human review instead of being silently treated as identical.
- Perceptual near-duplicate scanning remains a required gate because compact hashes can collide and exact hashes cannot detect every re-encoding.

## Portable image-manifest adapter

`load_manifest_image_corpus` converts an approved manifest into reusable `ImageExample` splits. Paths must be relative to an explicit media root and cannot escape it. Every file is rehashed before admission, and the path-free audit records the manifest digest plus split, label, generator-family, source-license, and complete attribution summaries.

Generator-holdout rows are exposed by normalized family name, while the base manifest validator guarantees that those families are absent from training, validation, and calibration. This makes adding another approved local generator a data-only operation rather than another hard-coded dataset adapter.

For matched evaluation, place one real control and one generated derivative in the same `content_group` and `generator_holdout` split. Their semantic categories must agree. `matched_generator_holdouts` rejects missing or multiply represented controls and conflicting slice metadata instead of silently constructing an imbalanced or source-confounded comparison.

After freezing training, calibration, threshold, and abstention policy, evaluate every admitted family with:

```bash
forensic-evaluate-manifest-holdouts \
  --manifest data/local/manifest.csv \
  --media-root data/local/media \
  --checkpoint artifacts/spatial-frequency-v1.pt \
  --output reports/manifest-generator-holdouts.json
```

The report includes per-family metrics and content-group bootstrap intervals plus macro and worst-family AUROC. It also evaluates matched-control-source, semantic-category, and capture-device slices without fitting new thresholds. Overall and slice records include equal-mass reliability bins, Wilson intervals for sensitivity and specificity, and coverage/selective risk at the explicitly recorded abstention margin. Acceptance is ineligible unless slice metadata is complete and at least three held-out families each contain 1,000 matched content groups per class; small fixtures can validate mechanics but cannot pass by construction.
