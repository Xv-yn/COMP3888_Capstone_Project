"""
- Validate the directory structure.
- Write a *normalized* data YAML that works from any working directory:
      path: <absolute dataset root>
      train: train/images
      val:   val/images
      test:  test/images  (only if present)
      nc:    <class count>
      names: [<class0>, <class1>, ...]
  If <dataset_root>/data.yaml exists, copy class names from it when possible.
- Launch Ultralytics training and print locations of artifacts.
- ALSO: print live 'accuracy' each epoch (defined as mAP@0.50 IoU), plus mAP50-95, precision, recall.

Usage
-----
    python train.py /path/to/dataset \
        --weights weights/yolov8m.pt \
        --epochs 200 --batch 16 --imgsz 640 \
        --name exp1

Outputs
-------
Artifacts are written under:
    results/training/<name>/
      ├─ weights/{best.pt,last.pt}
      ├─ results.csv
      └─ (optional plots) labels.jpg, results.png, etc.
"""

import argparse
from pathlib import Path
import shutil
import sys

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def assert_yolo_dir(ds: Path):
    """
    Validate that dataset contains the expected YOLO directory layout.
    Exits the process with an error message if required paths are missing.
    """
    req = [
        ds / "train" / "images",
        ds / "train" / "labels",
        ds / "val" / "images",
        ds / "val" / "labels",
    ]
    for p in req:
        if not p.exists():
            sys.exit(f"[ERROR] Missing required path: {p}")

    # sanity: must contain at least one image in train/val
    has_train_imgs = any((ds / "train" / "images").rglob(f"*{ext}") for ext in IMG_EXTS)
    has_val_imgs = any((ds / "val" / "images").rglob(f"*{ext}") for ext in IMG_EXTS)
    if not (has_train_imgs and has_val_imgs):
        sys.exit("[ERROR] No images found under train/images or val/images.")


def write_min_yaml(ds: Path, out_yaml: Path):
    """
    If ds/data.yaml exists, read nc/names from it, but always normalize paths so the file
    works from anywhere:
        path: <abs_ds>
        train: train/images
        val: val/images
        test: test/images (only if it exists)
    If no data.yaml, infer classes from labels and write minimal YAML.
    """
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    # Try to pull nc/names from existing data.yaml (if present)
    existing_yaml = ds / "data.yaml"
    nc, names = None, None
    if existing_yaml.exists():
        try:
            # Prefer PyYAML if available
            import yaml  # type: ignore

            with open(existing_yaml, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            # Accept both forms: names: list or names: {idx: name}
            names_field = d.get("names", None)
            if isinstance(names_field, dict):
                # convert dict to list in index order
                idxs = sorted(int(k) for k in names_field.keys())
                names = [names_field[str(i)] for i in idxs]
            elif isinstance(names_field, list):
                names = names_field
            nc = d.get("nc", len(names) if names else None)
        except Exception:
            # Fall back to infer below
            pass

    # Fallback: infer class count from label files if nc/names unknown
    if nc is None or names is None:
        label_dir = ds / "train" / "labels"
        classes = set()
        if label_dir.exists():
            for lf in label_dir.rglob("*.txt"):
                for line in lf.read_text().splitlines():
                    if line.strip():
                        parts = line.split()
                        if parts:
                            try:
                                classes.add(int(parts[0]))
                            except ValueError:
                                pass
        if classes:
            nc = max(classes) + 1
            names = [f"class_{i}" for i in range(nc)]
        else:
            # Safe default so training can at least start
            nc = 1
            names = ["object"]

    # test/images only if present
    test_rel = "test/images" if (ds / "test" / "images").exists() else None

    # Write normalized YAML
    lines = [
        f"path: {ds.as_posix()}",
        "train: train/images",
        "val: val/images",
    ]
    if test_rel:
        lines.append("test: test/images")
    lines += [
        f"nc: {int(nc)}",
        f"names: {names}",
    ]
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote normalized dataset YAML to: {out_yaml}")
    return out_yaml


# ----------------------- Live metric printing (accuracy) -----------------------
def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def make_epoch_printer():
    """
    Returns a callback that prints 'accuracy' (mAP50) and other key metrics
    at the end of each epoch during Ultralytics training.
    """
    state = {"best_map50": 0.0}

    def on_fit_epoch_end(trainer):
        # 'trainer.metrics' is a dict populated after validation each epoch.
        # Keys commonly include: 'metrics/precision(B)', 'metrics/recall(B)',
        # 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'. Fall back to plain keys if needed.
        m = getattr(trainer, "metrics", {}) or {}

        p = m.get("metrics/precision(B)", m.get("precision"))
        r = m.get("metrics/recall(B)", m.get("recall"))
        m50 = m.get("metrics/mAP50(B)", m.get("map50"))
        m5095 = m.get("metrics/mAP50-95(B)", m.get("map"))

        if isinstance(m50, (int, float)) and m50 > state["best_map50"]:
            state["best_map50"] = float(m50)

        print(
            f"[METRICS] acc(mAP50)={_fmt(m50)}  mAP50-95={_fmt(m5095)}  "
            f"P={_fmt(p)}  R={_fmt(r)}  best_acc={_fmt(state['best_map50'])}"
        )

    return on_fit_epoch_end


# ------------------------------------------------------------------------------


def parse_args():
    """
    CLI argument parsing.
    """
    ap = argparse.ArgumentParser(description="Train YOLOv8 on a standard YOLO dataset folder.")
    ap.add_argument("dataset_dir", type=str, help="Path to dataset directory (YOLO format).")
    ap.add_argument(
        "--weights",
        type=str,
        default="weights/yolov8m.pt",
        help="Initial weights (e.g., yolov8n.pt, weights/yolov8m.pt).",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None, help="'0', 'cpu', 'mps', etc.")
    ap.add_argument("--project", type=str, default="results/training")
    ap.add_argument("--name", type=str, default="run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    """
    Entry point: validate dataset, write YAML, run training, print artifact paths.
    Also prints live 'accuracy' (mAP50) each epoch via a callback.
    """
    args = parse_args()
    ds = Path(args.dataset_dir).expanduser().resolve()
    if not ds.exists() or not ds.is_dir():
        sys.exit(f"[ERROR] Dataset directory not found: {ds}")

    assert_yolo_dir(ds)

    # minimal data yaml (required by ultralytics)
    out_yaml = Path(args.project) / args.name / "dataset.yaml"
    data_yaml = write_min_yaml(ds, out_yaml)

    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Register live metric printer
    model.add_callback("on_fit_epoch_end", make_epoch_printer())

    print("[INFO] Starting training…")
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        save=True,
        save_period=-1,  # keep last + best
        pretrained=True,
        verbose=True,
    )

    out_dir = Path(args.project) / args.name
    print("\n[INFO] Training complete.")
    print(f"[INFO] Artifacts: {out_dir}")
    print(f"[INFO] Best checkpoint: {out_dir / 'weights' / 'best.pt'}")
    print(f"[INFO] Last checkpoint: {out_dir / 'weights' / 'last.pt'}")
    print(f"[INFO] Metrics CSV: {out_dir / 'results.csv'}")

    # Try to print a final 'accuracy' (mAP50) summary if available
    final_map50 = None
    try:
        # Some Ultralytics versions return a results object with .metrics dict
        final_map50 = getattr(results, "metrics", {}).get("metrics/mAP50(B)", None) or getattr(
            results, "metrics", {}
        ).get("map50", None)
    except Exception:
        pass

    if isinstance(final_map50, (int, float)):
        print(f"[INFO] Final accuracy (mAP50): {final_map50:.4f}")


if __name__ == "__main__":
    main()
