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
| `source_type` | Camera, curated, procedural, generated, or other documented type |
| `license_id` | Reviewed license/terms identifier |
| `generator_family` | Required for synthetic samples |
| `transformation` | Original or ordered processing lineage |

## Admission checklist

Before training, record the source terms, allowed research/commercial use, redistribution limits, attribution requirements, privacy/consent conditions, model-weight license, collection date, and reviewer decision outside the media directory. Unknown licenses are rejected.

## Leakage controls

- A file hash and sample ID are unique.
- All derivatives of a content identity stay in one split.
- Generator-holdout families are absent from train, validation, and calibration.
- Real-source/device holdouts follow the same rule when those fields are available.
- Raw collections are deduplicated before manifest creation with `partition_candidates`. Exact-byte groups are kept in one deterministic split, while perceptual-hash collisions are reported for human review instead of being silently treated as identical.
- Perceptual near-duplicate scanning remains a required gate because compact hashes can collide and exact hashes cannot detect every re-encoding.
