"""Configuration loading.

The experimental configuration lives in a YAML file so that every run can be
reproduced from a single artifact. Values are merged over :data:`DEFAULTS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data": {
        "image_size": 224,
        "color_space": "hsv",
    },
    "features": {
        "hist_bins": 16,
        "include_quality": True,
    },
    "model": {
        "kind": "logreg",
        "C": 1.0,
        "max_iter": 2000,
        "n_estimators": 300,
    },
    "predict": {
        "batch_size": 64,
    },
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the default config merged with the YAML file at *path*."""
    if path is None:
        return _deep_merge(DEFAULTS, {})
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config root must be a mapping, got {type(loaded).__name__}")
    return _deep_merge(DEFAULTS, loaded)
