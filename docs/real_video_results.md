# Real-video temporal baseline

The first real-video experiment trains `temporal-residual-v1` from scratch on DAVIS 2017 real sequences and GenVidBench Keling generated clips. It evaluates once on held-out DAVIS sequences and the unseen Sora generator family. The checkpoint is local and ignored because both training datasets are CC BY-NC 4.0; it is a non-commercial research artifact.

## Audited data

| Source | Role | Retained sources |
| --- | --- | ---: |
| DAVIS 2017 | real train, validation, and test control | 90 |
| GenVidBench Keling T2V | generated train and validation | 220 |
| GenVidBench Sora | unseen generated test | 48 |

The content-hash audit retained 358 independent sources across 233 train, 59 validation, and 66 test examples. Five long composite or promotional files were excluded before splitting. No identical content crosses splits.

Verified local archives:

- DAVIS 2017 480p: `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`
- GenVidBench Keling: `c596cc8f65752cb63a58b9cadaaf2395a8f73a547f6c99d1a2dbb89f6d2b1145`
- GenVidBench Sora: `fe7f41af1d1354ecb9d35f78081d896d12980a153a98570709800f2d4a29d739`

## Result

| Test slice | AUROC | Balanced accuracy | ECE |
| --- | ---: | ---: | ---: |
| Clean | 0.5752 | 0.4826 | 0.3480 |
| JPEG quality 30 | 0.5764 | 0.4826 | 0.3465 |
| Gaussian blur | 0.5787 | 0.4826 | 0.3336 |
| 50% resize | 0.5752 | 0.4826 | 0.3492 |
| Four-frame sparse sampling | 0.5729 | 0.5139 | 0.3469 |

The clean grouped-bootstrap 95% interval is 0.3941–0.7315. Validation AUROC reached 0.9309, but unseen-generator AUROC fell to 0.5752. This gap is the main learning result: the compact model learned Keling/DAVIS distinctions that largely failed to transfer to Sora/DAVIS.

Temporal evidence had a larger mean absolute logit contribution (1.7834) than frame appearance (1.0525), but that does not make it reliable—the held-out metrics show that the temporal contribution itself did not generalize adequately.

## Reproduction

Install `requirements-neural.lock`, place the credited datasets under ignored local data directories, and run:

```bash
PYTHONPATH=src python3 -m forensic_model.neural_video_experiment \
  --davis-root data/raw/davis2017/DAVIS \
  --keling-root data/raw/genvidbench/Keling \
  --sora-root data/raw/genvidbench/OpenAI_Sora \
  --output reports/real-video-evaluation.json \
  --checkpoint artifacts/temporal-residual-v1.pt
```

The committed report SHA-256 is `2351bdd74eefbd9f59f484f8589e3d1c567a92eae85907876c8e07413182e972`. The ignored checkpoint SHA-256 is `aa26276ee1901f0ddb6929b407f1b089e59b09f8f0ef10c21c5aaeb51b675e9e`.

## Limits and next experiment

Real and generated labels still come from different collections, codecs, durations, and content distributions. The result is diagnostic, not production assurance or generator attribution. The next experiment must add matched real/generated content controls, multiple training generators, a frame-aggregation baseline using the frozen image detector, and measured decode/inference latency. The temporal model must beat that matched baseline before the Phase 5 exit gate can close.
