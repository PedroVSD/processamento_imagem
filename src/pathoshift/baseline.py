"""Classical baseline: colour/texture descriptors + linear model.

This is the mandatory *simple and reproducible* comparator. It is deliberately
cheap so that inference over 2,250 images stays far below the 10-minute budget
on a 4-core CPU, and so that the CNN proposal has a fair reference point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from tqdm import tqdm

from .images import background_fraction, focus_score, load_from_manifest, to_hsv
from .manifests import resolve_path

EPS = 1e-8


@dataclass
class BaselineModel:
    """A fitted scikit-learn pipeline plus the metadata needed to reload it."""

    pipeline: Any
    labels: list[str]
    features: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: np.ndarray) -> tuple[list[str], np.ndarray]:
        return predict_with_pipeline(self.pipeline, self.labels, features)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a BaselineModel")
        return model


def _channel_histogram(channel: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(channel, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + EPS)


def extract_features(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Build a fixed-length descriptor from one RGB image.

    Composition: normalised HSV histograms (colour), per-channel statistics
    (stain intensity) and two quality descriptors (focus and background).
    """
    features_cfg = config.get("features", {})
    bins = int(features_cfg.get("hist_bins", 16))
    include_quality = bool(features_cfg.get("include_quality", True))

    hsv = to_hsv(image).astype(np.float32)
    parts = [
        _channel_histogram(hsv[..., 0], bins, (0, 180)),
        _channel_histogram(hsv[..., 1], bins, (0, 256)),
        _channel_histogram(hsv[..., 2], bins, (0, 256)),
        image.reshape(-1, 3).mean(axis=0) / 255.0,
        image.reshape(-1, 3).std(axis=0) / 255.0,
    ]
    if include_quality:
        parts.append(
            np.array(
                [np.log1p(focus_score(image)) / 10.0, background_fraction(image)],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts).astype(np.float32)


def build_features(
    frame,
    images_root: str | Path | None,
    config: Mapping[str, Any],
    show_progress: bool = True,
) -> np.ndarray:
    root = Path(images_root) if images_root else None
    size = int(config.get("data", {}).get("image_size", 224))
    rows: list[np.ndarray] = []
    iterator = frame.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(frame), desc="features", unit="img")
    for row in iterator:
        image = load_from_manifest(row.path, root, size=size)
        rows.append(extract_features(image, config))
    if not rows:
        raise ValueError("no images to featurise")
    return np.stack(rows)


def build_estimator(config: Mapping[str, Any]):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_cfg = config.get("model", {})
    seed = int(config.get("seed", 42))
    kind = str(model_cfg.get("kind", "logreg")).lower()

    if kind in {"rf", "random_forest", "randomforest"}:
        estimator = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            n_jobs=-1,
            random_state=seed,
        )
    elif kind in {"logreg", "logistic_regression", "lr"}:
        estimator = LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            random_state=seed,
        )
    else:
        raise ValueError(
            f"unknown model kind {kind!r}; the CNN baseline is not implemented yet "
            "(see README) — use 'logreg' or 'rf'"
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def train_baseline(
    frame,
    images_root: str | Path | None,
    config: Mapping[str, Any],
) -> BaselineModel:
    if "label" not in frame.columns or frame["label"].isna().any():
        raise ValueError("training manifest must contain labels for every image")
    matrix = build_features(frame, images_root, config)
    target = frame["label"].astype(str).to_numpy()
    estimator = build_estimator(config)
    estimator.fit(matrix, target)
    return BaselineModel(
        pipeline=estimator,
        labels=[str(value) for value in estimator.classes_],
        features={"dim": int(matrix.shape[1])},
        config=dict(config),
    )


def predict_with_pipeline(
    pipeline,
    labels: Sequence[str],
    matrix: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(matrix)
        indices = probabilities.argmax(axis=1)
        predictions = [labels[index] for index in indices]
        confidence = probabilities[np.arange(len(indices)), indices]
    else:  # pragma: no cover - defensive
        predictions = [str(value) for value in pipeline.predict(matrix)]
        confidence = np.ones(len(predictions), dtype=np.float32)
    return predictions, np.asarray(confidence, dtype=np.float32)


def predict_manifest(
    frame,
    images_root: str | Path | None,
    model: BaselineModel,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = build_features(frame, images_root, config)
    labels, confidence = model.predict(matrix)
    return [
        {
            "image_id": str(image_id),
            "predicted_label": str(label),
            "confidence": float(score),
        }
        for image_id, label, score in zip(frame["image_id"], labels, confidence)
    ]


def missing_images(frame, images_root: str | Path | None, limit: int = 10) -> list[str]:
    root = Path(images_root) if images_root else None
    missing = [row.path for row in frame.itertuples(index=False) if resolve_path(row.path, root) is None]
    return missing[:limit]
