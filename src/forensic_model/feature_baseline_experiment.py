"""Matched real-data experiment for the interpretable Phase 1 image baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import PIL
import torch

from forensic_model.feature_baseline import (
    FeatureTrainingConfig,
    extract_dataset_features,
    fit_feature_detector,
    score_feature_model,
)
from forensic_model.metrics import auroc, binary_metrics, grouped_bootstrap_interval
from forensic_model.neural_data import (
    SYNTHSCARS_ATTRIBUTION,
    PillowImageDataset,
    discover_cifake,
    discover_synthscars_test,
    evaluation_transform,
    image_split_digest,
)


def run_feature_baseline_experiment(
    cifake_root: Path,
    synthscars_root: Path,
    *,
    output: Path,
    artifact: Path,
    neural_report: Path | None = None,
    identity_policy: Path | None = None,
    training_config: FeatureTrainingConfig = FeatureTrainingConfig(),
) -> dict[str, object]:
    """Fit and evaluate the Phase 1 model on the neural model's frozen splits."""

    bundle = discover_cifake(cifake_root, identity_policy=identity_policy)
    training_features, training_labels, _ = extract_dataset_features(
        PillowImageDataset(bundle.train, evaluation_transform()),
        batch_size=training_config.batch_size,
    )
    validation_features, validation_labels, _ = extract_dataset_features(
        PillowImageDataset(bundle.validation, evaluation_transform()),
        batch_size=training_config.batch_size,
    )
    trained = fit_feature_detector(
        training_features,
        training_labels,
        validation_features,
        validation_labels,
        config=training_config,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    trained.detector.save(artifact)

    stress_metrics: dict[str, dict[str, object]] = {}
    clean_details: tuple[list[float], list[int], list[str], torch.Tensor] | None = None
    for operation in ("clean", "jpeg30", "blur1", "resize50"):
        features, labels, groups = extract_dataset_features(
            PillowImageDataset(bundle.test, evaluation_transform(operation=operation)),
            batch_size=training_config.batch_size,
        )
        probabilities = _calibrated_probabilities(trained.detector, features)
        stress_metrics[operation] = binary_metrics(
            labels,
            probabilities,
            threshold=trained.detector.policy.threshold,
        ).to_dict()
        if operation == "clean":
            clean_details = probabilities, labels, groups, features
    if clean_details is None:
        raise RuntimeError("clean feature evaluation was not produced")
    clean_probabilities, clean_labels, clean_groups, clean_features = clean_details
    interval = grouped_bootstrap_interval(
        clean_labels,
        clean_probabilities,
        clean_groups,
        statistic=auroc,
        resamples=500,
        seed=training_config.seed,
    )

    synthscars = discover_synthscars_test(synthscars_root)
    real_rows = tuple(example for example in bundle.test if example.label == 0)[: len(synthscars)]
    unseen_features, unseen_labels, _ = extract_dataset_features(
        PillowImageDataset(real_rows + synthscars, evaluation_transform()),
        batch_size=training_config.batch_size,
    )
    unseen_probabilities = _calibrated_probabilities(trained.detector, unseen_features)
    unseen_metrics = binary_metrics(
        unseen_labels,
        unseen_probabilities,
        threshold=trained.detector.policy.threshold,
    )
    report: dict[str, object] = {
        "claim_scope": "matched_phase_1_research_baseline_not_production_assurance",
        "training": "from_scratch",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pillow": PIL.__version__,
            "device": "cpu",
        },
        "configuration": asdict(training_config),
        "model": {
            "architecture": "standardized_logistic_regression",
            "feature_version": trained.detector.model.feature_version,
            "parameters": len(trained.detector.model.weights) + 1,
            "pretrained_weights": False,
            "artifact_sha256": _sha256(artifact),
        },
        "data_audit": asdict(bundle.audit),
        "split_membership_sha256": image_split_digest(bundle),
        "attribution": [asdict(item) for item in bundle.attributions + (SYNTHSCARS_ATTRIBUTION,)],
        "final_training_loss": trained.final_training_loss,
        "calibration": trained.detector.calibrator.to_dict() if trained.detector.calibrator else None,
        "threshold_selected_on_validation": trained.detector.policy.threshold,
        "cifake_test": stress_metrics,
        "clean_auroc_grouped_bootstrap_95_percent": asdict(interval),
        "unseen_synthscars_with_cifake_real_control": unseen_metrics.to_dict(),
        "unseen_evaluation_limit": (
            "SynthScars contains synthetic-only holdout images; pairing with CIFAKE real images introduces content, "
            "resolution, and source confounds, so this metric is diagnostic rather than a deployment claim."
        ),
        "explanation_summary": _contribution_summary(trained.detector.model, clean_features[:256]),
    }
    if neural_report is not None:
        report["comparison_to_neural"] = _compare_neural(report, neural_report)
    if bundle.identity_policy is not None:
        report["identity_policy"] = asdict(bundle.identity_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-train-feature-baseline")
    parser.add_argument("--cifake-root", type=Path, required=True)
    parser.add_argument("--synthscars-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--neural-report", type=Path)
    parser.add_argument("--identity-policy", type=Path)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    options = parser.parse_args(arguments)
    run_feature_baseline_experiment(
        options.cifake_root,
        options.synthscars_root,
        output=options.output,
        artifact=options.artifact,
        neural_report=options.neural_report,
        identity_policy=options.identity_policy,
        training_config=FeatureTrainingConfig(
            epochs=options.epochs,
            batch_size=options.batch_size,
            cpu_threads=options.cpu_threads,
            seed=options.seed,
        ),
    )
    return 0


def _calibrated_probabilities(detector, features: torch.Tensor) -> list[float]:
    if detector.calibrator is None:
        raise ValueError("feature detector must be calibrated")
    return [detector.calibrator.transform(score) for score in score_feature_model(detector.model, features)]


def _contribution_summary(model, features: torch.Tensor) -> dict[str, object]:
    means = torch.tensor(model.means, dtype=torch.float64)
    scales = torch.tensor(model.scales, dtype=torch.float64)
    weights = torch.tensor(model.weights, dtype=torch.float64)
    standardized = ((features.to(torch.float64) - means) / scales).clamp(
        -model.standardized_clip,
        model.standardized_clip,
    )
    contributions = standardized * weights
    return {
        "sample_count": len(features),
        "mean_absolute_contribution": {
            name: float(contributions[:, index].abs().mean())
            for index, name in enumerate(model.feature_names)
        },
    }


def _compare_neural(feature_report: dict[str, Any], neural_report_path: Path) -> dict[str, object]:
    neural = json.loads(neural_report_path.read_text(encoding="utf-8"))
    if neural.get("data_audit") != feature_report["data_audit"]:
        raise ValueError("neural comparison report does not use the identical CIFAKE audit")
    if neural.get("split_membership_sha256") != feature_report["split_membership_sha256"]:
        raise ValueError("neural comparison report does not use identical CIFAKE split membership")
    feature_slices = feature_report["cifake_test"]
    neural_slices = neural["cifake_test"]
    operations = ("clean", "jpeg30", "blur1", "resize50")
    return {
        "neural_report_sha256": _sha256(neural_report_path),
        "neural_minus_feature_auroc": {
            operation: float(neural_slices[operation]["auroc"] - feature_slices[operation]["auroc"])
            for operation in operations
        },
        "same_data_audit": True,
        "same_split_membership": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
