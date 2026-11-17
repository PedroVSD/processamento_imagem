"""Local evaluator: mirrors the official ``evaluate_pathoshift_crc.mjs``.

The official evaluator checks the prediction contract and reports accuracy,
macro F1, per-class F1 and the confusion matrix. This module does the same in
Python so experiments can be iterated without a Node.js round-trip, and adds
the validation-versus-robustness comparison required by the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import LABELS
from .manifests import ManifestError


def _labels_present(y_true: Sequence[str], y_pred: Sequence[str]) -> list[str]:
    observed = set(y_true) | set(y_pred)
    return [label for label in LABELS if label in observed]


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidence: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro F1, per-class metrics and the confusion matrix."""
    y_true = [str(value).strip().upper() for value in y_true]
    y_pred = [str(value).strip().upper() for value in y_pred]
    if len(y_true) != len(y_pred):
        raise ManifestError(
            f"length mismatch: {len(y_true)} ground truth vs {len(y_pred)} predictions"
        )
    if not y_true:
        raise ManifestError("no samples to evaluate")

    unknown = sorted((set(y_true) | set(y_pred)) - set(LABELS))
    if unknown:
        raise ManifestError(f"labels outside the official nine classes: {unknown}")

    labels = _labels_present(y_true, y_pred)
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)

    accuracy = float((true_array == pred_array).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        true_array, pred_array, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float(
        np.average(f1, weights=support) if support.sum() else 0.0
    )

    matrix = confusion_matrix(true_array, pred_array, labels=labels)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "labels": labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
    }

    if confidence is not None:
        confidences = np.asarray([float(value) for value in confidence])
        correct = true_array == pred_array
        metrics["mean_confidence"] = float(confidences.mean())
        metrics["mean_confidence_correct"] = (
            float(confidences[correct].mean()) if correct.any() else None
        )
        metrics["mean_confidence_wrong"] = (
            float(confidences[~correct].mean()) if (~correct).any() else None
        )
        bin_index = np.clip((confidences * 10).astype(int), 0, 9)
        bins = []
        for bin_id in range(10):
            mask = bin_index == bin_id
            if not mask.any():
                continue
            bins.append(
                {
                    "bin": f"{bin_id / 10:.1f}-{(bin_id + 1) / 10:.1f}",
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidences[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
        metrics["reliability"] = bins

    return metrics


def compare(validation: Mapping[str, Any], robustness: Mapping[str, Any]) -> dict[str, Any]:
    """Report the macro-F1 drop between validation and robustness."""
    base = float(validation["macro_f1"])
    robust = float(robustness["macro_f1"])
    delta = robust - base
    per_class_delta = {}
    for label, values in validation["per_class"].items():
        robust_values = robustness["per_class"].get(label)
        if robust_values is None:
            continue
        per_class_delta[label] = {
            "validation_f1": values["f1"],
            "robustness_f1": robust_values["f1"],
            "delta_f1": robust_values["f1"] - values["f1"],
        }
    return {
        "validation_macro_f1": base,
        "robustness_macro_f1": robust,
        "delta_macro_f1": delta,
        "relative_drop": (delta / base) if base else None,
        "per_class_delta": per_class_delta,
    }


def format_table(metrics: Mapping[str, Any]) -> str:
    """Render a fixed-width per-class table for the terminal."""
    lines = [
        f"{'label':<6} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}",
        "-" * 44,
    ]
    for label in metrics["labels"]:
        values = metrics["per_class"][label]
        lines.append(
            f"{label:<6} {values['precision']:>10.4f} {values['recall']:>8.4f} "
            f"{values['f1']:>8.4f} {values['support']:>8d}"
        )
    lines.append("-" * 44)
    lines.append(
        f"{'macro':<6} {'':>10} {'':>8} {metrics['macro_f1']:>8.4f} "
        f"{metrics['n_samples']:>8d}"
    )
    lines.append(f"accuracy: {metrics['accuracy']:.4f}")
    return "\n".join(lines)


def save_metrics(metrics: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    return destination


def plot_confusion_matrix(
    metrics: Mapping[str, Any],
    output: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = metrics["confusion_matrix_labels"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title(f"{title} (macro F1={metrics['macro_f1']:.4f})")
    figure.tight_layout()

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a predictions CSV and validate the submission contract."""
    frame = pd.read_csv(path, dtype={"image_id": str})
    required = {"image_id", "predicted_label", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ManifestError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ManifestError(f"{path}: empty predictions file")

    frame["predicted_label"] = frame["predicted_label"].astype(str).str.strip().str.upper()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="raise")
    unknown = sorted(set(frame["predicted_label"]) - set(LABELS))
    if unknown:
        raise ManifestError(f"{path}: labels outside the nine classes: {unknown}")
    if ((frame["confidence"] < 0) | (frame["confidence"] > 1)).any():
        raise ManifestError(f"{path}: confidence outside [0, 1]")
    if frame["image_id"].duplicated().any():
        raise ManifestError(f"{path}: duplicated image_id")
    return frame


def evaluate_predictions(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Join predictions to ground truth, checking full coverage."""
    if "label" not in manifest.columns or manifest["label"].isna().all():
        raise ManifestError("manifest has no labels; pass --labels for evaluation")

    merged = manifest.loc[:, ["image_id", "label"]].merge(
        predictions, on="image_id", how="left", validate="one_to_one"
    )
    missing = int(merged["predicted_label"].isna().sum())
    if missing:
        raise ManifestError(f"{missing} manifest images have no prediction")

    return classification_metrics(
        merged["label"].astype(str).tolist(),
        merged["predicted_label"].astype(str).tolist(),
        merged["confidence"].astype(float).tolist(),
    )
