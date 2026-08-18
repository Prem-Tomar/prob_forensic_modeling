# Calibrated and explainable confidence

Raw classifier scores are not deployment probabilities. `ImageDetector.fit_calibrator` fits Platt scaling on a partition that was not used for model fitting or architecture selection. The detector preserves both the raw score and calibrated probability.

`ImageDetector.analyze_image` returns a typed result with:

- the operational decision, including `abstain`;
- calibrated probability of the synthetic class;
- decision confidence and frozen threshold;
- calibration method;
- the largest signed feature contributions;
- the complete contribution map for audit.

The abstention band is an operational policy, not a statistical confidence interval. Feature contributions exactly decompose the linear model score, but they explain model behavior rather than media authorship. Calibration and threshold parameters must be fitted on the calibration split and frozen before any test evaluation.
