#!/usr/bin/env python3
"""
YOLOv8 Testing Script — Dual-Threshold Evaluation

Acceptance Rate = (# cows ≥ stricter threshold) / (# cows ≥ baseline 0.5)

Usage:
    python yolov8_test.py --root ./testing_set --conf 0.8
    python yolov8_test.py --root ./testing_set --conf 0.85 --limit 20
    python yolov8_test.py --root ./testing_set --conf 0.8 --device cuda
"""

from time import perf_counter
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from cow_detectection.modeling.yolov8.predict import run_inference, draw_yolov8_results

# --- Fixed config (resolved relative to this file) ---
HERE        = Path(__file__).resolve().parent
WEIGHTS     = HERE.parent / "modeling" / "yolov8" / "weights" / "yolov8m.pt"
OUTPUT_DIR  = HERE / "yolo_results"
OUTPUT_PLOT = HERE / "yolo_reports"
OUTPUT_CSV  = HERE / OUTPUT_PLOT / "yolo_acceptance.csv"

TARGET_CLASS = "cow"
BASELINE_THR = 0.5

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def main(root_folder: str, conf_thr: float, device: str = "cpu", limit: int = 0):
    # Prepare I/O
    root = Path(root_folder).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = [p for p in root.rglob("*") if p.suffix.lower() in VALID_EXT]
    images.sort()
    if limit and limit > 0:
        images = images[:limit]
    n_imgs = len(images)
    if n_imgs == 0:
        print(f"No images found under {root}")
        return

    # Load model
    model = YOLO(str(WEIGHTS))  # device selection is handled inside ultralytics; pass --device if you use CLI
    print(f"Testing {n_imgs} images (YOLOv8 cow detection) on device={device}")
    print(f"Stricter confidence threshold = {conf_thr}")
    print(f"Baseline threshold = {BASELINE_THR}\n")

    total_baseline_cows = 0
    total_passed_cows = 0

    rates_per_image = []
    base_counts = []
    strict_counts = []

    # Optional CSV rows
    csv_rows = [["image", "num_cows_baseline", "num_cows_strict", "accept_rate_image"]]

    t0 = perf_counter()
    for idx, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        t1 = perf_counter()

        # --- Single inference at BASELINE_THR ---
        det_base = run_inference(model, frame, conf_thr=BASELINE_THR)

        # Keep only 'cow' (case-insensitive, defensively)
        cows_base = []
        for d in det_base:
            name = str(d.get("cls_name", "")).lower()
            if name == TARGET_CLASS:
                cows_base.append(d)

        # Baseline denominator
        num_base = len(cows_base)

        # Numerator (stricter): count cows whose score >= conf_thr
        num_strict = 0
        for d in cows_base:
            # Try 'score' or 'conf' key; default to 0.0
            score = float(d.get("score", d.get("conf", 0.0)))
            if score >= conf_thr:
                num_strict += 1

        # Running totals & per-image stats
        total_baseline_cows += num_base
        total_passed_cows += num_strict
        img_rate = (num_strict / num_base) if num_base > 0 else np.nan

        base_counts.append(num_base)
        strict_counts.append(num_strict)
        rates_per_image.append(img_rate)

        dt_ms = (perf_counter() - t1) * 1000.0
        running_rate = (100.0 * total_passed_cows / total_baseline_cows) if total_baseline_cows > 0 else 0.0

        print(f"[{idx:03d}/{n_imgs:03d}] {img_path.name:<25} "
              f"≥{conf_thr:.2f}: {num_strict:2d} / ≥{BASELINE_THR:.1f}: {num_base:2d} "
              f"| time: {dt_ms:.1f} ms | running acc: {running_rate:.1f}%")

        # Save visualization of *baseline* detections (≥ 0.5)
        vis = frame.copy()
        draw_yolov8_results(vis, cows_base)
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), vis)

        # CSV row
        csv_rows.append([str(img_path.relative_to(root)), num_base, num_strict,
                         (f"{img_rate:.3f}" if np.isfinite(img_rate) else "")])

    elapsed = perf_counter() - t0
    final_rate = (100.0 * total_passed_cows / total_baseline_cows) if total_baseline_cows > 0 else 0.0

    print("\n===== YOLOv8 Cow Detection Acceptance (Summary) =====")
    print(f"Processed {n_imgs} images in {elapsed:.1f}s")
    print(f"Total cows (≥ {BASELINE_THR}): {total_baseline_cows}")
    print(f"Cows (≥ {conf_thr}): {total_passed_cows}")
    print(f"Acceptance rate (≥{conf_thr} / ≥{BASELINE_THR}): {final_rate:.2f}%")
    print(f"Annotated images (showing ≥{BASELINE_THR}) saved to: {OUTPUT_DIR}\n")

    # Save CSV
    with OUTPUT_CSV.open("w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[OK] Per-image CSV → {OUTPUT_CSV}")

    # --- Plot metrics ---

    # Histogram of per-image acceptance rates
    finite_rates = [r for r in rates_per_image if np.isfinite(r)]
    plt.figure(figsize=(6, 4))
    plt.hist(finite_rates, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Per-Image Acceptance Rate (≥{conf_thr}/≥{BASELINE_THR})")
    plt.xlabel("Acceptance rate per image")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT / "acceptance_hist.png", dpi=170)
    print(f"[INFO] Saved acceptance histogram → {OUTPUT_PLOT / 'acceptance_hist.png'}")

    # Threshold curve using the first image’s detections at baseline
    # (Counts for thresholds ≥ 0.5 can be derived by thresholding the same list)
    if n_imgs > 0:
        first_frame = cv2.imread(str(images[0]))
        det_first = run_inference(model, first_frame, conf_thr=BASELINE_THR)
        cows_first = [d for d in det_first if str(d.get("cls_name","")).lower() == TARGET_CLASS]
        scores_first = [float(d.get("score", d.get("conf", 0.0))) for d in cows_first]

        thresholds = np.linspace(0.5, 0.95, 10)
        counts = [(np.array(scores_first) >= thr).sum() for thr in thresholds]

        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, counts, marker="o")
        plt.title("Cows detected vs. threshold (example image)")
        plt.xlabel("Confidence threshold")
        plt.ylabel("# cows detected (≥ threshold)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT / "threshold_curve.png", dpi=170)
        print(f"[INFO] Saved threshold curve → {OUTPUT_PLOT / 'threshold_curve.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with test images (recursively scanned)")
    ap.add_argument("--conf", type=float, default=0.8, help="stricter confidence threshold for numerator")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="device hint (for info only)")
    ap.add_argument("--limit", type=int, default=0, help="only process first N images")
    args = ap.parse_args()
    main(args.root, args.conf, device=args.device, limit=args.limit)
