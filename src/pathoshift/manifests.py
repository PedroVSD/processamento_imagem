"""Reading, normalising and validating the official manifests.

The exact column names of ``train.csv`` / ``validation.csv`` / ``robustness.csv``
are fixed by the discipline package, but they are not known to us at
development time.  Instead of hard-coding them, the loaders detect the id, path
and label columns through a list of candidate names and normalise everything to
a small internal schema.

Internal manifest schema
------------------------
image_id : str
    Stable identifier; echoes the value that must appear in the predictions CSV.
path : str
    Path as written in the manifest (may be relative to ``--images-root``).
label : str or NA
    One of :data:`pathoshift.LABELS`, when the manifest carries labels.

Internal robustness schema
--------------------------
image_id : str
kind : str
    Normalised perturbation name (``color``, ``jpeg``, ``blur`` or ``noise``).
params : dict
    Numeric parameters found in the remaining columns.
seed : int
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from . import LABELS
from .perturbations import normalise_kind

ID_CANDIDATES: Sequence[str] = (
    "image_id",
    "imageid",
    "id",
    "image",
    "image_name",
    "filename",
    "file_name",
    "file",
    "name",
    "tile_id",
)
PATH_CANDIDATES: Sequence[str] = (
    "path",
    "filepath",
    "file_path",
    "image_path",
    "img_path",
    "relpath",
    "relative_path",
    "image",
    "filename",
    "file_name",
)
LABEL_CANDIDATES: Sequence[str] = (
    "label",
    "class",
    "category",
    "target",
    "tumor_type",
    "tissue",
    "diagnosis",
)
KIND_CANDIDATES: Sequence[str] = (
    "kind",
    "type",
    "perturbation",
    "transform",
    "augmentation",
    "operation",
    "corruption",
)
SEED_CANDIDATES: Sequence[str] = ("seed", "rng_seed", "random_state")
COLUMN_CANDIDATES: Sequence[str] = (
    "column",
    "target_column",
    "field",
    "channel",
    "axis",
)


class ManifestError(ValueError):
    """Raised when a manifest violates the expected contract."""


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower()


def _find_column(columns: Iterable[Any], candidates: Sequence[str]) -> Any | None:
    lookup = {_normalise_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV manifest, sniffing the delimiter."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    try:
        frame = pd.read_csv(manifest_path, sep=None, engine="python")
    except pd.errors.ParserError as exc:  # pragma: no cover - malformed input
        raise ManifestError(f"could not parse {manifest_path}: {exc}") from exc
    if frame.empty:
        raise ManifestError(f"manifest is empty: {manifest_path}")
    return frame


def _has_known_column(frame: pd.DataFrame) -> bool:
    known = set(ID_CANDIDATES) | set(PATH_CANDIDATES) | set(LABEL_CANDIDATES)
    columns = {_normalise_name(column) for column in frame.columns}
    return bool(columns & known)


def _recover_headerless(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Re-read a manifest that has no header row.

    Some manifests are plain lists (one path per line, or ``path,label``).
    We detect that case by checking whether no candidate column name appears,
    and then assume positional columns.
    """
    raw = pd.read_csv(path, sep=None, engine="python", header=None)
    width = raw.shape[1]
    if width == 1:
        names = ["path"]
    elif width == 2:
        names = ["image_id", "label"]
    else:
        names = ["image_id", "path", "label"][:width]
    raw.columns = names
    return raw


def read_manifest(path: str | Path, strict_labels: bool = True) -> pd.DataFrame:
    """Read a manifest into the internal schema ``image_id, path, label``."""
    manifest_path = Path(path)
    frame = _read_csv(manifest_path)
    if not _has_known_column(frame):
        frame = _recover_headerless(frame, manifest_path)

    id_column = _find_column(frame.columns, ID_CANDIDATES)
    path_column = _find_column(frame.columns, PATH_CANDIDATES)
    label_column = _find_column(frame.columns, LABEL_CANDIDATES)

    if path_column is None and id_column is None:
        raise ManifestError(
            f"{manifest_path}: no image id or path column found "
            f"(columns: {list(frame.columns)})"
        )

    normalised = pd.DataFrame(index=frame.index)
    if id_column is not None:
        normalised["image_id"] = frame[id_column].astype(str).str.strip()
    elif path_column is not None:
        normalised["image_id"] = frame[path_column].astype(str).map(
            lambda value: Path(value).stem
        )
    else:  # pragma: no cover - guarded above
        raise ManifestError(f"{manifest_path}: cannot build an image_id")

    if path_column is not None:
        normalised["path"] = frame[path_column].astype(str).str.strip()
    else:
        normalised["path"] = normalised["image_id"]

    if label_column is not None:
        labels = frame[label_column].astype(str).str.strip().str.upper()
        normalised["label"] = labels
        if strict_labels:
            unknown = sorted(set(labels) - set(LABELS))
            if unknown:
                raise ManifestError(
                    f"{manifest_path}: labels outside the official nine classes: {unknown}"
                )
    else:
        normalised["label"] = pd.NA

    normalised = normalised.reset_index(drop=True)
    duplicates = normalised["image_id"].duplicated(keep=False)
    if duplicates.any():
        sample = normalised.loc[duplicates, "image_id"].head(5).tolist()
        raise ManifestError(
            f"{manifest_path}: duplicate image_id values, e.g. {sample}"
        )
    return normalised


def read_robustness(path: str | Path) -> pd.DataFrame:
    """Read ``robustness.csv`` into ``image_id, kind, params, seed``."""
    manifest_path = Path(path)
    frame = _read_csv(manifest_path)

    id_column = _find_column(frame.columns, ID_CANDIDATES)
    kind_column = _find_column(frame.columns, KIND_CANDIDATES)
    seed_column = _find_column(frame.columns, SEED_CANDIDATES)
    if id_column is None:
        raise ManifestError(f"{manifest_path}: no image id column found")
    if kind_column is None:
        raise ManifestError(f"{manifest_path}: no perturbation column found")

    structural = {id_column, kind_column, seed_column}
    parameter_columns = [column for column in frame.columns if column not in structural]

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        params: dict[str, Any] = {}
        for column in parameter_columns:
            value = row[column]
            if pd.isna(value):
                continue
            column_key = _normalise_name(column)
            if column_key in {c for c in COLUMN_CANDIDATES}:
                params["column"] = str(value).strip().lower()
                continue
            params[column_key] = _coerce_number(value)
        seed = 0
        if seed_column is not None and not pd.isna(row[seed_column]):
            seed = int(_coerce_number(row[seed_column]))
        records.append(
            {
                "image_id": str(row[id_column]).strip(),
                "kind": normalise_kind(row[kind_column]),
                "params": params,
                "seed": seed,
            }
        )
    return pd.DataFrame(records)


def _coerce_number(value: Any) -> Any:
    """Return *value* as int/float when possible, otherwise as stripped text."""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def validate_images(
    frame: pd.DataFrame,
    images_root: str | Path | None = None,
    sample: int | None = 20,
) -> dict[str, Any]:
    """Check that the referenced images exist and report missing entries."""
    root = Path(images_root) if images_root else None
    missing: list[str] = []
    checked = 0
    for path in frame["path"]:
        if sample is not None and checked >= sample:
            break
        checked += 1
        if resolve_path(path, root) is None:
            missing.append(str(path))
    return {
        "checked": checked,
        "missing": missing,
        "n_missing_sampled": len(missing),
    }


def resolve_path(raw_path: str, images_root: Path | None = None) -> Path | None:
    """Resolve a manifest path, tolerating relative/absolute variants."""
    candidate = Path(str(raw_path))
    options: list[Path] = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        if images_root is not None:
            options.append(Path(images_root) / candidate)
            options.append(Path(images_root) / candidate.name)
        options.append(candidate)
        options.append(candidate.name)
    for option in options:
        if option.is_file():
            return option
    return None


def merge_labels(frame: pd.DataFrame, labels_path: str | Path) -> pd.DataFrame:
    """Attach labels from ``labels.csv`` when the manifest has none."""
    label_frame = read_manifest(labels_path)
    if label_frame["label"].isna().all():
        raise ManifestError(f"{labels_path}: no label column found")
    merged = frame.drop(columns=["label"], errors="ignore").merge(
        label_frame[["image_id", "label"]], on="image_id", how="left"
    )
    if merged["label"].isna().any():
        missing = int(merged["label"].isna().sum())
        raise ManifestError(f"{missing} images have no label after merging {labels_path}")
    return merged


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].dropna().value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def write_manifest(frame: pd.DataFrame, output: str | Path) -> Path:
    """Persist a manifest as CSV (or JSONL when the suffix says so)."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jsonl", ".json"}:
        records = frame.to_dict(orient="records")
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    else:
        frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def write_predictions(
    frame: pd.DataFrame,
    output: str | Path,
    ordered_ids: Sequence[str] | None = None,
) -> Path:
    """Write ``image_id,predicted_label,confidence`` exactly as required."""
    required = {"image_id", "predicted_label", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ManifestError(f"predictions missing columns: {sorted(missing)}")

    predictions = frame.loc[:, ["image_id", "predicted_label", "confidence"]].copy()
    predictions["image_id"] = predictions["image_id"].astype(str)
    predictions["predicted_label"] = predictions["predicted_label"].astype(str).str.upper()
    predictions["confidence"] = pd.to_numeric(predictions["confidence"], errors="raise")

    unknown = sorted(set(predictions["predicted_label"]) - set(LABELS))
    if unknown:
        raise ManifestError(f"predictions outside the nine classes: {unknown}")
    out_of_range = predictions[
        (predictions["confidence"] < 0.0) | (predictions["confidence"] > 1.0)
    ]
    if not out_of_range.empty:
        raise ManifestError(
            f"{len(out_of_range)} confidences outside [0, 1], e.g. "
            f"{out_of_range.head(3).to_dict(orient='records')}"
        )
    if predictions["image_id"].duplicated().any():
        raise ManifestError("duplicate image_id in predictions")

    if ordered_ids is not None:
        expected = list(ordered_ids)
        if len(expected) != len(predictions) or set(expected) != set(predictions["image_id"]):
            raise ManifestError(
                "predictions do not match the manifest ids "
                f"(manifest={len(expected)}, predictions={len(predictions)})"
            )
        predictions = predictions.set_index("image_id").loc[expected].reset_index()

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False, encoding="utf-8")
    return destination


def iterations(frame: pd.DataFrame) -> Iterator[tuple[str, str, Any]]:
    for row in frame.itertuples(index=False):
        yield row.image_id, row.path, getattr(row, "label", pd.NA)
