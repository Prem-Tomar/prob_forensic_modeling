# Image MVP

## What it teaches

The first model is standardized logistic regression over ten deterministic image summaries. It makes the entire learning loop inspectable: decoding, feature versioning, fitting, serialization, probability scoring, and per-feature contributions.

The summaries cover channel means, luminance variance, saturation, horizontal/vertical residuals, Laplacian energy, checkerboard energy, and clipped channels. These signals are deliberately simple. They establish an auditable baseline and can reveal dataset shortcuts; they are not expected to solve unseen-generator detection alone.

## Library usage

```python
from forensic_model import ImageDetector, RGBImage

training_images = [RGBImage.from_rows(rows) for rows in training_rows]
detector = ImageDetector.train(training_images, labels=[0, 1, 0, 1])
result = detector.predict_image(candidate)

print(result.probability_synthetic)
print(result.contributions)
```

`ImageDetector.predict_file` accepts an `ImageDecoder` implementation. The dependency-free `PPMDecoder` supports deterministic fixtures. Applications can supply a codec adapter for PNG, JPEG, WebP, or other formats without coupling the learned model to a specific image library.

## Interpretation

The returned number is an uncalibrated probability-like model output until the calibration phase is fitted on a separate calibration split. Feature contributions explain the linear score, not the cause or authorship of an image. The library must abstain or qualify output when later scope and calibration checks are unavailable.

## Limitations

- The smoke tests use procedural patterns and validate mechanics only.
- Summary features discard spatial detail and are vulnerable to content and processing shortcuts.
- PPM fixtures do not represent deployment codecs.
- No real-world metric is reported until approved local datasets satisfy the evaluation contract.
