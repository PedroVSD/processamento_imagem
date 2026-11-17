"""Deterministic robustness perturbations.

``robustness.csv`` applies one of four perturbations per validation image:
colour/brightness change, JPEG compression, Gaussian blur or Gaussian noise.
The parameters and the seed of each case come from the manifest, so the
transformation must be a pure function of ``(image, kind, params, seed)``.

Parameter names in the manifest are not known in advance; every parameter is
looked up through a list of aliases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

SUPPORTED_KINDS: tuple[str, ...] = ("color", "jpeg", "blur", "noise")

_ALIASES: dict[str, str] = {
    # colour / brightness / contrast
    "colour": "color",
    "brightness": "color",
    "contrast": "color",
    "brightness_color": "color",
    "color_jitter": "color",
    "colour_jitter": "color",
    "hue": "color",
    "saturation": "color",
    "photometric": "color",
    "intensity": "color",
    # jpeg compression
    "jpg": "jpeg",
    "compression": "jpeg",
    "jpeg_compression": "jpeg",
    "compress": "jpeg",
    "quality": "jpeg",
    # blur
    "gaussian_blur": "blur",
    "gaussianblur": "blur",
    "gauss_blur": "blur",
    "defocus": "blur",
    "blurring": "blur",
    # noise
    "gaussian_noise": "noise",
    "gauss_noise": "noise",
    "white_noise": "noise",
    "additive_noise": "noise",
}

GAIN_KEYS = ("factor", "alpha", "gain", "brightness", "scale", "beta_scale")
OFFSET_KEYS = ("delta", "beta", "offset", "bias", "add")
QUALITY_KEYS = ("quality", "q", "jpeg_quality")
SIGMA_KEYS = ("sigma", "std", "stddev", "std_dev", "radius", "scale", "level")
MEAN_KEYS = ("mean", "mu", "loc")


def normalise_kind(kind: Any) -> str:
    """Map a manifest perturbation name to a supported kind."""
    text = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _ALIASES.get(text, text)
    if canonical not in SUPPORTED_KINDS:
        raise ValueError(
            f"unsupported perturbation {kind!r}; expected one of {SUPPORTED_KINDS} "
            f"or a known alias ({sorted(_ALIASES)})"
        )
    return canonical


def case_seed(image_id: str, seed: int | None = None) -> int:
    """Stable per-image seed, so reruns reproduce identical noise."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _get(params: Mapping[str, Any], keys: tuple[str, ...], default: float | None) -> Any:
    normalised = {str(key).strip().lower(): value for key, value in params.items()}
    for key in keys:
        if key in normalised and not _is_null(normalised[key]):
            return normalised[key]
    return default


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number):
        return default
    return number


def _as_int(value: Any, default: int) -> int:
    return int(round(_as_float(value, float(default))))


def apply_perturbation(
    image: np.ndarray,
    kind: Any,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Apply a deterministic perturbation and return a ``uint8`` RGB image."""
    params = dict(params or {})
    canonical = normalise_kind(kind)
    if canonical == "color":
        return _apply_color(image, params)
    if canonical == "jpeg":
        return _apply_jpeg(image, params)
    if canonical == "blur":
        return _apply_blur(image, params)
    return _apply_noise(image, params, seed)


def apply_case(image: np.ndarray, row: Mapping[str, Any], global_seed: int = 0) -> np.ndarray:
    """Apply one row of the robustness manifest to *image*."""
    seed = row.get("seed") or 0
    image_seed = case_seed(str(row.get("image_id", "")), int(seed) or global_seed)
    return apply_perturbation(image, row.get("kind"), row.get("params") or {}, image_seed)


def _apply_color(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Per-channel gain and additive offset, with optional saturation change.

    An offset whose magnitude is at most 1 is interpreted as a fraction of the
    0-255 range, matching how manifests usually express brightness shifts.
    """
    gain = _as_float(_get(params, GAIN_KEYS, None), 1.0)
    offset = _as_float(_get(params, OFFSET_KEYS, None), 0.0)
    if 0 < abs(offset) <= 1.0:
        offset *= 255.0

    gains = np.array(
        [
            _as_float(params.get("r", params.get("red")), 1.0),
            _as_float(params.get("g", params.get("green")), 1.0),
            _as_float(params.get("b", params.get("blue")), 1.0),
        ],
        dtype=np.float32,
    ) * np.float32(gain)

    channel = str(params.get("column", "all")).strip().lower()
    channels = {"r": 0, "red": 0, "g": 1, "green": 1, "b": 2, "blue": 2}
    target = channels.get(channel)
    if target is not None:
        masked = np.ones(3, dtype=np.float32)
        masked[target] = gains[target]
        gains = masked

    rgb = image.astype(np.float32) * gains + offset

    saturation = _get(params, ("saturation", "sat"), None)
    if saturation is not None:
        hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * _as_float(saturation, 1.0), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _apply_jpeg(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    quality = _as_int(_get(params, QUALITY_KEYS, None), 50)
    quality = int(np.clip(quality, 1, 100))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - encoder failure
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _apply_blur(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    sigma = _as_float(_get(params, SIGMA_KEYS, None), 1.0)
    if sigma <= 0:
        return image.copy()
    kernel = _get(params, ("ksize", "kernel", "kernel_size", "size"), None)
    if kernel is not None:
        ksize = _as_int(kernel, 0)
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(ksize, 1)
    else:
        ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def _apply_noise(image: np.ndarray, params: Mapping[str, Any], seed: int) -> np.ndarray:
    sigma = _as_float(_get(params, SIGMA_KEYS, None), 0.05)
    if 0 < sigma <= 1.0:
        sigma *= 255.0
    mean = _as_float(_get(params, MEAN_KEYS, None), 0.0)
    if abs(mean) <= 1.0 and mean != 0.0:
        mean *= 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def summarise_cases(frame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in frame["kind"]:
        canonical = normalise_kind(kind)
        counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def write_perturbed_image(image: np.ndarray, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(destination),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )
    return destination
