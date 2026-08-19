# From-scratch neural training

The optional neural track uses `SpatialFrequencyDetector`, a compact two-branch convolutional model. One branch learns spatial texture evidence; the other learns log-frequency evidence. Its final linear classifier is deliberately additive, so every logit can be decomposed into spatial and frequency contributions.

`discover_cifake` hashes all 120,000 source files, reports perceptual collisions, removes exact duplicates, and assigns new deterministic train, validation, and test splits. This is necessary because the downloaded CIFAKE release contains exact duplicates across its published train and test folders. The split seed is part of the experiment configuration.

`train_neural_detector` initializes every parameter from the recorded seed. It never downloads or loads pretrained weights. Optimization uses only the training split. The real-image experiment compares fixed spatial/frequency blends across clean, JPEG, blur, and resize validation slices, chooses the blend with the strongest worst-case ranking, then refits probability calibration and the decision threshold on clean validation scores. Test and generator-holdout samples remain untouched until every choice is fixed.

The checkpoint retains both learned branches and records the selected frequency weight. This keeps the training ablation inspectable while ensuring library inference, explanations, and the video frame baseline use the same deployed blend.

`CalibratedNeuralImageDetector` is the reusable checkpoint boundary. It accepts local files or preprocessed tensor batches and returns a calibrated decision, abstention state, threshold, confidence, and both raw and deployed branch contributions. It does not download weights or media.

The neural extra is intentionally optional:

```bash
python -m pip install -e '.[neural]'
```

Applications may use `PillowImageDataset` with their own licensed `ImageExample` records. Raw media, checkpoints, and local environments are ignored by Git. Published checkpoints must record the dataset manifest digest, split seed, dependency versions, training configuration, and code revision.

## Dataset credit

- CIFAKE: Bird and Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, IEEE Access, 2024. Dataset terms: MIT.
- CIFAR-10 real-image source: Krizhevsky and Hinton, *Learning Multiple Layers of Features from Tiny Images*, 2009. Terms stated by the CIFAKE release: MIT.
- SynthScars holdout: Kang et al., *LEGION: Learning to Ground and Explain for Synthetic Image Detection*, 2025. Dataset terms: Apache-2.0.

These datasets are downloaded from their official locations and are not redistributed by this library.
