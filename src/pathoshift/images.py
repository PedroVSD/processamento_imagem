"""Image loading and basic quality measurements.

Every read goes through :func:`load_rgb` so that dtypes, channel order and
resizing stay identical between training, validation and robustness inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from .manifests import resolve_path

Image.MAX_IMAGE_PIXELS = None  # tiles are small; avoid spurious decompression bombs


def load_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an image as ``uint8`` RGB, optionally resized to ``size x size``."""
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if size is not None and image.size != (size, size):
            image = image.resize((size, size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(array)


def load_from_manifest(
    raw_path: str,
    images_root: Path | None,
    size: int | None = None,
) -> np.ndarray:
    resolved = resolve_path(raw_path, images_root)
    if resolved is None:
        raise FileNotFoundError(
            f"image not found: {raw_path!r} (images_root={images_root})"
        )
    return load_rgb(resolved, size=size)


def to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def focus_score(image: np.ndarray) -> float:
    """Variance of the Laplacian: low values indicate blurry tiles."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def background_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that look like empty slide background (low chroma)."""
    hsv = to_hsv(image)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    background = (saturation < 0.10) & (value > 0.80)
    return float(background.mean())


def describe_image(image: np.ndarray) -> dict[str, float | int]:
    hsv = to_hsv(image)
    lab = to_lab(image)
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "mean_r": float(image[..., 0].mean()),
        "mean_g": float(image[..., 1].mean()),
        "mean_b": float(image[..., 2].mean()),
        "std_gray": float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).std()),
        "mean_saturation": float(hsv[..., 1].mean()),
        "mean_value": float(hsv[..., 2].mean()),
        "mean_lab_a": float(lab[..., 1].mean()),
        "mean_lab_b": float(lab[..., 2].mean()),
        "focus_laplacian": focus_score(image),
        "background_fraction": background_fraction(image),
    }


def iter_manifest_images(
    frame,
    images_root: Path | None,
    size: int | None = None,
):
    """Yield ``(image_id, image)`` for a manifest frame."""
    for row in frame.itertuples(index=False):
        yield row.image_id, load_from_manifest(row.path, images_root, size=size)


def ensure_dir(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_contact_sheet(
    images: Sequence[tuple[str, np.ndarray]],
    output: str | Path,
    columns: int = 4,
    tile: int = 224,
) -> Path:
    """Write a labelled contact sheet, useful for the error gallery."""
    if not images:
        raise ValueError("no images to plot")
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * tile, columns * tile, 3), 255, dtype=np.uint8)
    for index, (caption, image) in enumerate(images):
        row, column = divmod(index, columns)
        resized = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        sheet[row * tile : (row + 1) * tile, column * tile : (column + 1) * tile] = resized
        cv2.putText(
            sheet,
            caption[:28],
            (column * tile + 4, row * tile + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return destination


def stack_images(paths: Iterable[Path], size: int) -> np.ndarray:
    return np.stack([load_rgb(path, size=size) for path in paths])
