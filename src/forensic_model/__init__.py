"""Public library API for learned forensic modeling."""

from forensic_model.calibration import PlattCalibrator
from forensic_model.decision import DecisionPolicy, DetectionResult, Reason
from forensic_model.detector import ImageDetector
from forensic_model.features import FEATURE_NAMES, FEATURE_VERSION, FeatureVector, extract_features
from forensic_model.image import ImageDecodeError, ImageDecoder, PPMDecoder, RGBImage
from forensic_model.manifest import ManifestError, Sample, load_manifest, manifest_digest, validate_manifest
from forensic_model.metrics import BinaryMetrics, ConfidenceInterval, auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.model import LogisticModel, Prediction
from forensic_model.provenance import (
    ForensicResult,
    ProvenanceEvidence,
    ProvenanceStatus,
    ProvenanceVerifier,
    check_provenance,
    fuse_evidence,
)
from forensic_model.robustness import (
    EvaluationSample,
    Transformation,
    default_transformations,
    evaluate_postprocessing,
    evaluate_samples,
    evaluate_unseen_generators,
)
from forensic_model.video import (
    TEMPORAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_VERSION,
    FrameScore,
    TimedFrame,
    VideoClip,
    VideoDecoder,
    VideoDetector,
    VideoResult,
    extract_temporal_features,
    sample_frames,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "TEMPORAL_FEATURE_NAMES",
    "TEMPORAL_FEATURE_VERSION",
    "FeatureVector",
    "ForensicResult",
    "FrameScore",
    "BinaryMetrics",
    "ConfidenceInterval",
    "EvaluationSample",
    "ImageDecodeError",
    "ImageDecoder",
    "ImageDetector",
    "LogisticModel",
    "ManifestError",
    "PPMDecoder",
    "Prediction",
    "PlattCalibrator",
    "ProvenanceEvidence",
    "ProvenanceStatus",
    "ProvenanceVerifier",
    "RGBImage",
    "Reason",
    "Sample",
    "Transformation",
    "TimedFrame",
    "VideoClip",
    "VideoDecoder",
    "VideoDetector",
    "VideoResult",
    "DecisionPolicy",
    "DetectionResult",
    "extract_features",
    "extract_temporal_features",
    "auroc",
    "binary_metrics",
    "check_provenance",
    "fuse_evidence",
    "default_transformations",
    "evaluate_postprocessing",
    "evaluate_samples",
    "evaluate_unseen_generators",
    "grouped_bootstrap_interval",
    "load_manifest",
    "manifest_digest",
    "sample_frames",
    "validate_manifest",
]

__version__ = "0.1.0"
