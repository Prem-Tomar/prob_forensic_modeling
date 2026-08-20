# CIFAKE perceptual-collision review

The CIFAKE adapter uses a 64-bit difference hash as a screening signal, never as automatic proof that two files share an identity. The reproducible review utility enumerated all exact difference-hash collisions, measured normalized pixel distances, and rendered an ignored local contact sheet for visual inspection.

## Findings

- 120,000 candidate files were screened.
- 18 compact-hash collision groups produced 24 distinct-file pair comparisons.
- Two cross-label pairs were compact-hash collisions only: their grayscale mean absolute errors were 0.2410–0.2836, and visual content was unrelated.
- One same-label pair passed the conservative 0.02 grayscale-distance screen with error 0.0123.
- Visual inspection shows that low-distance pair is the same base scene with a red/blue color alteration.
- Both members of that pair deterministically map to the training partition under `cifake-split-v1`; it does not create current validation or test leakage.

The reviewed pair is identified without filesystem paths by hashes `f7fa19d4533d…a45f8` and `fda4c28a93c6…a0020`. The aggregate report SHA-256 is `8767ebd260929a2dc4ae945386d5af5e7803f7edecba83da51285040697614a6`.

## Reproduction

```bash
PYTHONPATH=src python -m forensic_model.near_duplicate_review \
  --cifake-root data/raw/cifake/DATASET \
  --output reports/cifake-collision-review.json \
  --contact-sheet artifacts/cifake-collision-contact-sheet.png \
  --thumbnail-size 64 --low-distance-threshold 0.02
```

The committed JSON contains aggregate measurements and content hashes but no local paths or media. The contact sheet remains ignored because it contains dataset thumbnails.

## Open data-policy decision

The current frozen experiment keeps the two files as distinct exact identities. Because both happen to be in training, this does not invalidate the published test results, but it can overweight one semantic scene during training. Future partitions should either:

1. assign both files one reviewed content identity while retaining both transformed observations; or
2. retain one canonical observation and exclude the alternate color version.

Those policies teach different models: grouping preserves transformation diversity, while exclusion removes duplicate weighting. Changing the policy changes the frozen split-membership contract and requires a new model/report generation rather than silently rewriting existing evidence.

Both resolutions are represented explicitly by the `reviewed-collision-policy-v1` JSON schema. `group_sha256` lists arrays of two or more exact hashes that should retain separate observations under one reviewed content identity; `exclude_sha256` lists observations to omit. A hash may receive only one decision, referenced hashes must exist, and grouped observations must share one label.

Pass the reviewed file through `--identity-policy` when training the feature and neural models and when running the frozen stress matrix. The policy digest is recorded in both experiment reports and in neural checkpoint metadata. No policy is applied by default, so selecting a resolution remains an explicit data-governance decision rather than a hidden preprocessing heuristic.
