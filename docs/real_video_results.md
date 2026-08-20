# Real-video temporal baseline

The real-video experiment trains `temporal-residual-v1` from scratch on DAVIS 2017 real sequences and both GenVidBench Keling text-to-video and image-to-video clips. It evaluates once on held-out DAVIS sequences and the unseen Sora generator family. The checkpoint is local and ignored because both training datasets are CC BY-NC 4.0; it is a non-commercial research artifact.

## Audited data

| Source | Role | Retained sources |
| --- | --- | ---: |
| DAVIS 2017 | real train, validation, and test control | 90 |
| GenVidBench Keling T2V | generated train and validation | 220 |
| GenVidBench Keling I2V | generated train and validation | 47 |
| GenVidBench Sora | unseen generated test | 48 |

The content-hash audit retained 405 independent sources across 270 train, 69 validation, and 66 test examples. Five long composite or promotional files were excluded before splitting. No identical content crosses splits. T2V and I2V increase generation-mode diversity, but both use the same Keling generator family; Sora remains the only unseen generator.

Verified local archives:

- DAVIS 2017 480p: `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`
- GenVidBench Keling: `c596cc8f65752cb63a58b9cadaaf2395a8f73a547f6c99d1a2dbb89f6d2b1145`
- GenVidBench Sora: `fe7f41af1d1354ecb9d35f78081d896d12980a153a98570709800f2d4a29d739`

## Result

| Test slice | AUROC | Balanced accuracy | ECE |
| --- | ---: | ---: | ---: |
| Clean | 0.6273 | 0.6181 | 0.1691 |
| JPEG quality 30 | 0.6262 | 0.6181 | 0.1686 |
| Gaussian blur | 0.6273 | 0.6181 | 0.1692 |
| 50% resize | 0.6262 | 0.6181 | 0.1690 |
| Four-frame sparse sampling | 0.6377 | 0.6007 | 0.1610 |

The clean grouped-bootstrap 95% interval is 0.4641–0.7674. Validation AUROC reached 0.8480, while unseen-generator AUROC reached 0.6273. Relative to T2V-only training, adding I2V improved unseen AUROC by 0.0521 and reduced ECE by 0.1789. Mode diversity helps, but the broad interval and low operating-point recall still show weak Sora transfer.

Frame appearance now has a larger mean absolute logit contribution (1.3133) than temporal evidence (0.8092). This reversal is consistent with reduced reliance on the brittle temporal component, but held-out ranking remains insufficient.

## Matched frame-aggregation comparison

The frozen CIFAKE image detector was applied to the same eight uniformly sampled frames per validation and test clip. Only clip-level calibration and threshold selection used the video validation split.

| Model | Test AUROC | 95% grouped interval | Balanced accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Frozen image-frame mean | 0.7442 | 0.6137–0.8675 | 0.7083 | 0.1919 |
| Temporal residual model | 0.6273 | 0.4641–0.7674 | 0.6181 | 0.1691 |

The temporal model trails the simpler frame mean by 0.1169 AUROC. It therefore still fails the Phase 5 adoption gate, although the deficit is 0.0521 smaller than under T2V-only training. On the 842.83-second generated Sora subset, frozen-frame decoding and inference took 101.31 seconds (7.21 seconds per input minute); the temporal model took 101.16 seconds (7.20 seconds per input minute). Similar timings show that video decoding dominates both compact models in this CPU environment.

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

The committed report SHA-256 is `ce5885f6dec0a66996e30ff9a276531ef6453f76d0cc34741686456ecb529566`. The ignored checkpoint SHA-256 is `c2f33b9f80c5b29b334d98ee6a63faeccfb5e736459bee089d8079ef3fd71eb3`.

Reproduce the matched frame baseline with:

```bash
PYTHONPATH=src python3 -m forensic_model.neural_video_baseline_experiment \
  --davis-root data/raw/davis2017/DAVIS \
  --keling-root data/raw/genvidbench/Keling \
  --sora-root data/raw/genvidbench/OpenAI_Sora \
  --image-checkpoint artifacts/spatial-frequency-v1.pt \
  --output reports/frame-aggregation-evaluation.json
```

The frame-aggregation report SHA-256 is `eee2dacd201e746e61b8d12b14a40d1e0ed4b0ed689f208688d220ce72a2f675`. Its frozen image checkpoint has raw SHA-256 `fd9b220f746a5e06c6c5b866797b605acd6cd10d5778fb66fec5bacba101642a` and semantic SHA-256 `9c5356f8f5db28b4015774521b5a8043e6cb78cba4a6267445f453f6eedf5783`.

## Limits and next experiment

Real and generated labels still come from different collections, codecs, durations, and content distributions. The result is diagnostic, not production assurance or generator attribution. The local corpus has only one training generator family despite its two modes. The next temporal experiment therefore still requires matched real/generated content controls and at least one additional training generator, then must beat the frozen frame-aggregation baseline before the Phase 5 exit gate can close.
