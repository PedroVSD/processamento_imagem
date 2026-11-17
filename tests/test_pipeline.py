"""End-to-end smoke tests with synthetic tiles.

These do not replace experiments on the official dataset: they only guarantee
that the manifest contract, the perturbations and the metric computation behave
as specified, so the pipeline can be trusted the day the real package arrives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathoshift import LABELS, metrics, perturbations
from pathoshift.cli import main
from pathoshift.manifests import ManifestError, read_manifest, read_robustness

CLASSES = ["ADI", "NORM", "TUM"]


def _make_tiles(root: Path, per_class: int = 6, size: int = 32) -> Path:
    rng = np.random.default_rng(0)
    for index, label in enumerate(CLASSES):
        folder = root / "images" / label
        folder.mkdir(parents=True, exist_ok=True)
        for number in range(per_class):
            base = np.full((size, size, 3), 40 + index * 60, dtype=np.uint8)
            noise = rng.integers(0, 25, size=(size, size, 3), dtype=np.uint8)
            image = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            from PIL import Image

            Image.fromarray(image).save(folder / f"{label}_{number:03d}.tif")
    return root / "images"


def _manifest(root: Path, name: str, per_class: int = 6) -> Path:
    rows = []
    for label in CLASSES:
        for number in range(per_class):
            rows.append(
                {
                    "image_id": f"{label}_{number:03d}",
                    "path": f"{label}/{label}_{number:03d}.tif",
                    "label": label,
                }
            )
    path = root / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def dataset(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path
    images_root = _make_tiles(root)
    return {
        "root": root,
        "images": images_root,
        "train": _manifest(root, "train.csv"),
        "validation": _manifest(root, "validation.csv", per_class=3),
    }


def test_read_manifest_contract(dataset):
    frame = read_manifest(dataset["train"])
    assert list(frame.columns) == ["image_id", "path", "label"]
    assert set(frame["label"]).issubset(set(LABELS))


def test_read_manifest_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"image_id": "a", "path": "a.tif", "label": "XYZ"}]).to_csv(path, index=False)
    with pytest.raises(ManifestError):
        read_manifest(path)


def test_perturbations_are_deterministic_and_bounded():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    case = {"image_id": "A_1", "kind": "noise", "params": {"sigma": 0.1}, "seed": 7}
    first = perturbations.apply_case(image, case, global_seed=42)
    second = perturbations.apply_case(image, case, global_seed=42)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.shape == image.shape


def test_perturbation_aliases_and_errors():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    darkened = perturbations.apply_perturbation(image, "brightness", {"factor": 0.5})
    assert darkened.mean() < image.mean()
    with pytest.raises(ValueError):
        perturbations.normalise_kind("sepia")


def test_read_robustness_normalises_params(tmp_path: Path):
    path = tmp_path / "robustness.csv"
    pd.DataFrame(
        [
            {"image_id": "TUM_1", "perturbation": "JPEG Compression", "quality": 30, "seed": 11},
            {"image_id": "ADI_1", "perturbation": "gaussian blur", "sigma": 1.5, "seed": 12},
        ]
    ).to_csv(path, index=False)
    frame = read_robustness(path)
    assert frame["kind"].tolist() == ["jpeg", "blur"]
    assert frame.loc[0, "params"]["quality"] == 30
    assert frame.loc[1, "seed"] == 12


def test_metrics_match_manual_computation():
    y_true = ["TUM", "TUM", "ADI", "NORM"]
    y_pred = ["TUM", "ADI", "ADI", "ADI"]
    result = metrics.classification_metrics(y_true, y_pred, [0.9, 0.6, 0.7, 0.5])
    assert result["accuracy"] == pytest.approx(0.5)
    # ADI: precision 1/3, recall 1.0 -> F1 0.5; TUM: precision 1.0, recall 0.5
    assert result["per_class"]["ADI"]["f1"] == pytest.approx(0.5, rel=1e-6)
    assert result["per_class"]["TUM"]["recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx((0.5 + 0.0 + 2 / 3) / 3, rel=1e-6)
    assert result["confusion_matrix_labels"] == ["ADI", "NORM", "TUM"]
    assert result["mean_confidence"] == pytest.approx(0.675)


def test_pipeline_end_to_end(dataset, tmp_path: Path):
    root = dataset["root"]
    outputs = tmp_path / "outputs"
    images_root = str(dataset["images"])

    assert main(["prepare", "--train", str(dataset["train"]), "--validation",
                 str(dataset["validation"]), "--images-root", images_root,
                 "--output", str(outputs / "manifests")]) == 0

    model = outputs / "baseline.joblib"
    assert main(["train", "--manifest", str(dataset["train"]), "--images-root", images_root,
                 "--output", str(model)]) == 0
    assert model.is_file()

    predictions = outputs / "predictions_validation.csv"
    assert main(["predict", "--manifest", str(dataset["validation"]),
                 "--images-root", images_root, "--model", str(model),
                 "--output", str(predictions)]) == 0

    frame = pd.read_csv(predictions)
    assert list(frame.columns) == ["image_id", "predicted_label", "confidence"]
    assert frame["confidence"].between(0, 1).all()

    metrics_path = outputs / "metrics.json"
    assert main(["evaluate", "--manifest", str(dataset["validation"]),
                 "--predictions", str(predictions), "--output", str(metrics_path)]) == 0
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "macro_f1" in payload["validation"]
    assert (outputs / "confusion_validation.png").is_file()
