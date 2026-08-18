"""Public library API for learned forensic modeling."""

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy, DetectionResult, Reason
from forensic_model.detector import ImageDetector
from forensic_model.features import FEATURE_NAMES, FEATURE_VERSION, FeatureVector, extract_features
from forensic_model.image import ImageDecodeError, ImageDecoder, PPMDecoder, RGBImage
from forensic_model.manifest import ManifestError, Sample, load_manifest, manifest_digest, validate_manifest
from forensic_model.model import LogisticModel, Prediction

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FeatureVector",
    "ImageDecodeError",
    "ImageDecoder",
    "ImageDetector",
    "LogisticModel",
    "ManifestError",
    "PPMDecoder",
    "Prediction",
    "PlattCalibrator",
    "RGBImage",
    "Reason",
    "Sample",
    "DecisionPolicy",
    "DetectionResult",
    "extract_features",
    "load_manifest",
    "manifest_digest",
    "validate_manifest",
]

__version__ = "0.1.0"
