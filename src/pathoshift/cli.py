"""Command line interface.

Usage mirrors the interface suggested by the discipline, but works directly
with the manifest CSVs:

    pathoshift describe
    pathoshift prepare   --train data/train.csv --validation data/validation.csv --output outputs/manifests
    pathoshift inspect   --manifest data/validation.csv --images-root data/images
    pathoshift train     --manifest data/train.csv --images-root data/images --output outputs/baseline.joblib
    pathoshift predict   --manifest data/validation.csv --images-root data/images --model outputs/baseline.joblib --output predictions_validation.csv
    pathoshift robustness --manifest data/validation.csv --robustness data/robustness.csv --images-root data/images --model outputs/baseline.joblib --output predictions_robustness.csv
    pathoshift evaluate  --manifest data/validation.csv --predictions predictions_validation.csv --robustness-predictions predictions_robustness.csv --output outputs/metrics.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import LABELS, __version__
from . import baseline, images, manifests, metrics, perturbations
from .config import load_config

DEFAULT_CONFIG = "configs/baseline.yaml"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_manifest(args, label_key: str = "labels") -> "Any":
    frame = manifests.read_manifest(args.manifest, strict_labels=False)
    labels_path = getattr(args, label_key, None)
    if labels_path and frame["label"].isna().any():
        frame = manifests.merge_labels(frame, labels_path)
    return frame


def _images_root(args) -> Path | None:
    root = getattr(args, "images_root", None)
    return Path(root) if root else None


def _report(frame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "images": int(len(frame)),
        "unique_ids": int(frame["image_id"].nunique()),
    }
    if "label" in frame.columns and frame["label"].notna().any():
        report["class_counts"] = manifests.class_distribution(frame)
    return report


def environment() -> dict[str, Any]:
    """Versions that make a run reproducible (no heavy imports)."""
    from importlib import metadata

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "pathoshift": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "tensorflow": "tensorflow",
    }
    for module, distribution in distributions.items():
        try:
            info[module] = metadata.version(distribution)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional
            info[module] = None
    return info


def _print(payload: Mapping[str, Any] | str) -> None:
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_describe(args) -> int:
    config = load_config(args.config)
    payload = {
        "environment": environment(),
        "labels": list(LABELS),
        "config": config,
        "config_path": str(args.config),
        "commands": [
            "prepare", "inspect", "train", "predict", "robustness", "evaluate",
        ],
    }
    _print(payload)
    return 0


def cmd_prepare(args) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for name in ("train", "validation"):
        path = getattr(args, name)
        if not path:
            continue
        frame = manifests.read_manifest(path, strict_labels=False)
        if args.labels:
            frame = manifests.merge_labels(frame, args.labels)
        checked = manifests.validate_images(frame, args.images_root, sample=args.validate_sample)
        destination = manifests.write_manifest(frame, output_dir / f"{name}.csv")
        summary[name] = {**_report(frame), "missing_sample": checked["missing"], "path": str(destination)}

    if args.robustness:
        robustness = manifests.read_robustness(args.robustness)
        destination = output_dir / "robustness.csv"
        robustness.assign(params=robustness["params"].astype(str)).to_csv(
            destination, index=False, encoding="utf-8"
        )
        summary["robustness"] = {
            "cases": int(len(robustness)),
            "kinds": perturbations.summarise_cases(robustness),
            "path": str(destination),
        }

    summary["environment"] = environment()
    summary["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    _print(summary)
    return 0


def cmd_inspect(args) -> int:
    frame = _load_manifest(args)
    root = _images_root(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = frame if args.sample <= 0 or args.sample >= len(frame) else frame.sample(
        n=args.sample, random_state=args.seed
    )

    rows: list[dict[str, Any]] = []
    sheet: list[tuple[str, np.ndarray]] = []
    from tqdm import tqdm

    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="inspect", unit="img"):
        try:
            image = images.load_from_manifest(row.path, root)
        except FileNotFoundError as exc:
            print(f"warning: {exc}", file=sys.stderr)
            continue
        stats = images.describe_image(image)
        stats["image_id"] = row.image_id
        stats["label"] = getattr(row, "label", None)
        rows.append(stats)
        if len(sheet) < args.sheet:
            sheet.append((f"{row.image_id} {getattr(row, 'label', '')}", image))

    if not rows:
        print("error: no image could be read; check --images-root", file=sys.stderr)
        return 2

    import pandas as pd

    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(output_dir / "image_stats.csv", index=False)

    numeric = stats_frame.select_dtypes(include=[np.number])
    summary = {
        "manifest": args.manifest,
        "images_total": int(len(frame)),
        "images_inspected": int(len(stats_frame)),
        "class_counts": manifests.class_distribution(frame),
        "dims": {
            "height": sorted(stats_frame["height"].unique().tolist()),
            "width": sorted(stats_frame["width"].unique().tolist()),
            "channels": sorted(stats_frame["channels"].unique().tolist()),
        },
        "quality": {
            column: {
                "mean": float(numeric[column].mean()),
                "p05": float(numeric[column].quantile(0.05)),
                "p50": float(numeric[column].quantile(0.50)),
                "p95": float(numeric[column].quantile(0.95)),
            }
            for column in numeric.columns
            if column not in {"height", "width", "channels"}
        },
        "environment": environment(),
    }
    with (output_dir / "inspect.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    if sheet:
        images.save_contact_sheet(sheet, output_dir / "contact_sheet.png")

    _print({key: summary[key] for key in ("images_total", "images_inspected", "class_counts", "dims")})
    print(f"wrote {output_dir / 'inspect.json'}")
    return 0


def _train_record(config_path: str, config: Mapping[str, Any], frame, model, elapsed: float) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": config_path,
        "config": config,
        "seed": int(config.get("seed", 42)),
        "n_train": int(len(frame)),
        "class_counts": manifests.class_distribution(frame),
        "feature_dim": model.features.get("dim"),
        "classes": model.labels,
        "train_seconds": round(elapsed, 2),
        "environment": environment(),
    }


def cmd_train(args) -> int:
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model_kind:
        config["model"]["kind"] = args.model_kind

    frame = _load_manifest(args)
    if frame["label"].isna().any():
        print("error: training manifest needs labels for every image", file=sys.stderr)
        return 2
    if args.limit:
        frame = frame.groupby("label", group_keys=False).head(
            max(1, args.limit // frame["label"].nunique())
        )

    started = time.perf_counter()
    model = baseline.train_baseline(frame, _images_root(args), config)
    elapsed = time.perf_counter() - started

    destination = model.save(args.output)
    record = _train_record(str(args.config), config, frame, model, elapsed)
    record_path = Path(args.output).with_suffix(".json")
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)

    _print(
        {
            "model": str(destination),
            "record": str(record_path),
            "n_train": record["n_train"],
            "train_seconds": record["train_seconds"],
        }
    )
    return 0


def _write_predictions(rows: Sequence[Mapping[str, Any]], output: str, frame) -> Path:
    import pandas as pd

    destination = manifests.write_predictions(
        pd.DataFrame(rows), output, ordered_ids=frame["image_id"].tolist()
    )
    print(f"wrote {destination} ({len(rows)} rows)")
    return destination


def cmd_predict(args) -> int:
    config = load_config(args.config)
    model = baseline.BaselineModel.load(args.model)
    model_config = model.config or config
    frame = _load_manifest(args)
    if args.limit:
        frame = frame.head(args.limit)

    started = time.perf_counter()
    rows = baseline.predict_manifest(frame, _images_root(args), model, model_config)
    print(f"inference: {time.perf_counter() - started:.1f}s for {len(rows)} images")
    _write_predictions(rows, args.output, frame)
    return 0


def cmd_robustness(args) -> int:
    config = load_config(args.config)
    model = baseline.BaselineModel.load(args.model)
    model_config = model.config or config

    frame = _load_manifest(args)
    cases = manifests.read_robustness(args.robustness)
    joined = frame.merge(cases, on="image_id", how="inner", validate="one_to_one")
    if joined.empty:
        print("error: no image_id matched between manifest and robustness.csv", file=sys.stderr)
        return 2
    if len(joined) != len(frame):
        print(
            f"warning: {len(frame) - len(joined)} manifest images have no robustness case",
            file=sys.stderr,
        )

    size = int(model_config.get("data", {}).get("image_size", 224))
    save_dir = Path(args.save_images) if args.save_images else None
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    from tqdm import tqdm

    started = time.perf_counter()
    for row in tqdm(joined.itertuples(index=False), total=len(joined), desc="robustness", unit="img"):
        image = images.load_from_manifest(row.path, _images_root(args), size=size)
        case = {"image_id": row.image_id, "kind": row.kind, "params": row.params, "seed": row.seed}
        perturbed = perturbations.apply_case(image, case, global_seed=int(model_config.get("seed", 42)))
        if save_dir is not None:
            perturbations.write_perturbed_image(perturbed, save_dir / f"{row.image_id}.png")
        features.append(baseline.extract_features(perturbed, model_config))

    matrix = np.stack(features)
    labels, confidence = model.predict(matrix)
    print(f"inference: {time.perf_counter() - started:.1f}s for {len(joined)} images")
    rows = [
        {"image_id": str(image_id), "predicted_label": str(label), "confidence": float(score)}
        for image_id, label, score in zip(joined["image_id"], labels, confidence)
    ]
    _write_predictions(rows, args.output, joined)
    return 0


def cmd_evaluate(args) -> int:
    frame = _load_manifest(args)
    predictions = metrics.load_predictions(args.predictions)
    result = metrics.evaluate_predictions(frame, predictions)
    print(metrics.format_table(result))

    payload: dict[str, Any] = {"validation": result}
    if args.robustness_predictions:
        robust_predictions = metrics.load_predictions(args.robustness_predictions)
        robust_frame = frame
        if args.robustness_manifest:
            robust_frame = manifests.read_manifest(args.robustness_manifest, strict_labels=False)
        robust = metrics.evaluate_predictions(robust_frame, robust_predictions)
        print("\nrobustness")
        print(metrics.format_table(robust))
        payload["robustness"] = robust
        payload["comparison"] = metrics.compare(result, robust)
        print(
            f"\nmacro F1: validation={result['macro_f1']:.4f} "
            f"robustness={robust['macro_f1']:.4f} "
            f"delta={payload['comparison']['delta_macro_f1']:+.4f}"
        )

    payload["environment"] = environment()
    output = Path(args.output)
    metrics.save_metrics(payload, output)
    print(f"wrote {output}")

    plot_dir = output.parent
    metrics.plot_confusion_matrix(result, plot_dir / "confusion_validation.png", "Validation")
    if "robustness" in payload:
        metrics.plot_confusion_matrix(
            payload["robustness"], plot_dir / "confusion_robustness.png", "Robustness"
        )
    if args.errors and not args.no_errors:
        _write_error_gallery(
            frame, predictions, Path(args.errors), args.errors_per_class, _images_root(args)
        )
    return 0


def _write_error_gallery(
    frame,
    predictions,
    output_dir: Path,
    per_class: int,
    root: Path | None = None,
) -> None:
    merged = frame.merge(predictions, on="image_id", how="inner")
    merged["correct"] = merged["label"] == merged["predicted_label"]
    errors = merged[~merged["correct"]]
    if errors.empty:
        print("no errors to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    errors.sort_values("confidence", ascending=False).to_csv(
        output_dir / "errors.csv", index=False
    )
    picked: list[tuple[str, np.ndarray]] = []
    for _, group in errors.groupby("label"):
        for row in group.head(per_class).itertuples(index=False):
            try:
                image = images.load_from_manifest(row.path, root)
            except FileNotFoundError:
                continue
            picked.append((f"{row.image_id} {row.label}->{row.predicted_label}", image))
    if picked:
        images.save_contact_sheet(picked, output_dir / "error_gallery.png")
        print(f"wrote {output_dir / 'error_gallery.png'}")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathoshift",
        description="PathoShift-CRC: robust histopathology patch pipeline",
    )
    parser.add_argument("--version", action="version", version=f"pathoshift {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub):
        sub.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
        sub.add_argument("--images-root", default=None, help="directory prepended to manifest paths")

    describe = subparsers.add_parser("describe", help="print environment, versions and config")
    describe.add_argument("--config", default=DEFAULT_CONFIG)
    describe.set_defaults(func=cmd_describe)

    prepare = subparsers.add_parser("prepare", help="validate manifests and write normalised copies")
    prepare.add_argument("--train", default=None)
    prepare.add_argument("--validation", default=None)
    prepare.add_argument("--robustness", default=None)
    prepare.add_argument("--labels", default=None, help="labels.csv with image_id,label")
    prepare.add_argument("--images-root", default=None)
    prepare.add_argument("--output", default="outputs/manifests")
    prepare.add_argument("--validate-sample", type=int, default=20, help="images to check per manifest")
    prepare.set_defaults(func=cmd_prepare)

    inspect = subparsers.add_parser("inspect", help="dataset and quality inspection")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--labels", default=None)
    inspect.add_argument("--images-root", default=None)
    inspect.add_argument("--output", default="outputs/eda")
    inspect.add_argument("--sample", type=int, default=300, help="0 inspects every image")
    inspect.add_argument("--sheet", type=int, default=12, help="images in the contact sheet")
    inspect.add_argument("--seed", type=int, default=42)
    inspect.set_defaults(func=cmd_inspect)

    train = subparsers.add_parser("train", help="train the classical baseline")
    add_config(train)
    train.add_argument("--manifest", required=True)
    train.add_argument("--labels", default=None)
    train.add_argument("--output", default="outputs/baseline.joblib")
    train.add_argument("--model-kind", default=None, help="logreg or rf")
    train.add_argument("--limit", type=int, default=0, help="subsample for smoke tests")
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(func=cmd_train)

    predict = subparsers.add_parser("predict", help="predict a manifest and write the CSV contract")
    add_config(predict)
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--labels", default=None)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", default="predictions_validation.csv")
    predict.add_argument("--limit", type=int, default=0)
    predict.set_defaults(func=cmd_predict)

    robustness = subparsers.add_parser("robustness", help="predict perturbed validation images")
    add_config(robustness)
    robustness.add_argument("--manifest", required=True)
    robustness.add_argument("--labels", default=None)
    robustness.add_argument("--robustness", required=True)
    robustness.add_argument("--model", required=True)
    robustness.add_argument("--output", default="predictions_robustness.csv")
    robustness.add_argument("--save-images", default=None, help="dump perturbed images for analysis")
    robustness.set_defaults(func=cmd_robustness)

    evaluate = subparsers.add_parser("evaluate", help="metrics, confusion matrix and error gallery")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--labels", default=None)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--robustness-manifest", default=None)
    evaluate.add_argument("--robustness-predictions", default=None)
    evaluate.add_argument("--output", default="outputs/metrics.json")
    evaluate.add_argument("--images-root", default=None)
    evaluate.add_argument("--errors", default="outputs/errors", help="error gallery directory")
    evaluate.add_argument("--no-errors", action="store_true", help="skip the error gallery")
    evaluate.add_argument("--errors-per-class", type=int, default=4)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
