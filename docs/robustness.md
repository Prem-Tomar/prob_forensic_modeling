# Robustness evaluation

The evaluation library computes ranking, decision, and calibration metrics without assuming row order. AUROC and average precision group tied scores. Expected calibration error uses equal-mass bins. Identity-group bootstrap intervals resample content groups rather than correlated derivative rows.

`evaluate_unseen_generators` requires the caller to declare generator families used for training and rejects overlap with evaluation families. Each unseen family is compared with the same real-media evaluation set and reported separately; macro aggregation belongs in the final report and must not hide a weak family.

`evaluate_postprocessing` applies deterministic decoded-pixel stresses and reports every slice alongside clean data. The built-in dependency-free matrix covers gamma shifts, blur, additive noise, quantization, and crop/rescale. Production evaluation must additionally use codec adapters for real JPEG/WebP recompression, resizing kernels, screenshots, and metadata operations as specified in the evaluation contract.

Synthetic tests prove metric and pipeline mechanics only. Acceptance gates require approved real data, generator-family holdouts, codec-level transformations, and versioned reports.
