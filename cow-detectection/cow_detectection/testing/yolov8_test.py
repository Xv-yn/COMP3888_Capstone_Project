#!/usr/bin/env python3
"""
YOLOv8 Testing Script — Dual-Threshold Evaluation

Acceptance Rate = (# cows ≥ stricter threshold) / (# cows ≥ baseline 0.5)

Usage:
    python yolov8_test.py --root ./testing_set --conf 0.8
"""

import time
from pathlib import Path
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

from cow_detectection.modeling.yolov8.predict import run_inference, draw_yolov8_results

# --- Fixed config ---
WEIGHTS = "../modeling/yolov8/weights/yolov8m.pt"
OUTPUT_DIR = "./yolo_results"
OUTPUT_PLOT = "./yolo_plots"
TARGET_CLASS = "cow"
BASELINE_THR = 0.5
# ---------------------

def main(root_folder, conf_thr):
    model = YOLO(WEIGHTS)
    root = Path(root_folder)
    out_dir = Path(OUTPUT_DIR)
    out_plot = Path(OUTPUT_PLOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
    image_paths = [p for p in root.rglob("*") if p.suffix.lower() in valid_ext]
    n_imgs = len(image_paths)
    if n_imgs == 0:
        print(f"No images found under {root}")
        return

    total_baseline_cows = 0
    total_passed_cows = 0

    print(f"Testing {n_imgs} images (YOLOv8 cow detection)...")
    print(f"Stricter confidence threshold = {conf_thr}")
    print(f"Baseline threshold = {BASELINE_THR}\n")

    t_start = time.time()
    rates_per_image = []
    base_counts = []
    strict_counts = []

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        t0 = time.time()

        # --- Denominator: baseline accepted cows (≥ 0.5) ---
        det_base = run_inference(model, frame, conf_thr=BASELINE_THR)
        cows_base = [d for d in det_base if str(d["cls_name"]).lower() == TARGET_CLASS]
        num_base = len(cows_base)

        # --- Numerator: stricter passed cows (≥ conf_thr) ---
        det_strict = run_inference(model, frame, conf_thr=conf_thr)
        cows_strict = [d for d in det_strict if str(d["cls_name"]).lower() == TARGET_CLASS]
        num_strict = len(cows_strict)

        total_baseline_cows += num_base
        total_passed_cows += num_strict

        running_rate = (100.0 * total_passed_cows / total_baseline_cows) if total_baseline_cows > 0 else 0.0
        dt_ms = (time.time() - t0) * 1000.0

        base_counts.append(num_base)
        strict_counts.append(num_strict)
        rates_per_image.append(num_strict / num_base if num_base > 0 else np.nan)


        print(f"[{idx:03d}/{n_imgs:03d}] {img_path.name:<25} "
              f"≥{conf_thr}: {num_strict:2d} / ≥{BASELINE_THR}: {num_base:2d} "
              f"| time: {dt_ms:.1f} ms | running acc: {running_rate:.1f}%")

        # ✅ Save visualization showing *all cows ≥ 0.5* (baseline detections)
        vis = frame.copy()
        draw_yolov8_results(vis, cows_base)  # changed from cows_strict → cows_base
        cv2.imwrite(str(out_dir / img_path.name), vis)

    elapsed = time.time() - t_start
    final_rate = (100.0 * total_passed_cows / total_baseline_cows) if total_baseline_cows > 0 else 0.0

    print("\n===== YOLOv8 Cow Detection Acceptance (Summary) =====")
    print(f"Processed {n_imgs} images in {elapsed:.1f}s")
    print(f"Total cows (≥ {BASELINE_THR}): {total_baseline_cows}")
    print(f"Cows (≥ {conf_thr}): {total_passed_cows}")
    print(f"Acceptance rate (≥{conf_thr} / ≥{BASELINE_THR}): {final_rate:.2f}%")
    print(f"Annotated images (showing ≥{BASELINE_THR}) saved to: {out_dir}\n")

    # --- Plot metrics ---
    plt.figure(figsize=(6,4))
    plt.hist([r for r in rates_per_image if not np.isnan(r)], bins=20, color="skyblue", edgecolor="black")
    plt.title("Per-Image Acceptance Rate (≥{}/≥{})".format(conf_thr, BASELINE_THR))
    plt.xlabel("Acceptance rate per image")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot / "acceptance_hist.png")
    print(f"[INFO] Saved acceptance histogram → {out_dir/'acceptance_hist.png'}")

    # --- Optional: threshold curve ---
    thresholds = np.linspace(0.5, 0.95, 10)
    ratios = []
    for thr in thresholds:
        det_high = run_inference(model, cv2.imread(str(image_paths[0])), conf_thr=thr)
        cows_high = [d for d in det_high if d["cls_name"].lower() == TARGET_CLASS]
        ratios.append(len(cows_high))
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, ratios, marker="o")
    plt.title("Cows detected vs. threshold")
    plt.xlabel("Confidence threshold")
    plt.ylabel("# cows detected (example image)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( out_plot / "threshold_curve.png")
    print(f"[INFO] Saved threshold curve → {out_dir/'threshold_curve.png'}")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with test images")
    ap.add_argument("--conf", type=float, default=0.8, help="stricter confidence threshold for numerator")
    args = ap.parse_args()
    main(args.root, args.conf)
