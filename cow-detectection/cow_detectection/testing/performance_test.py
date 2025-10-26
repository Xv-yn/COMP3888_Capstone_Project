#!/usr/bin/env python3
"""
performance_test.py  —  Multi-stage Livestock Analytics Benchmark
YOLOv8 → HRNet → TS-STG with accurate timing, CSV export, and plots.

Run examples:
    python performance_test.py --root ./testing_set
    python performance_test.py --root ./testing_set --device cuda --limit 20 --no-skeleton
"""

from __future__ import annotations
import csv
from time import perf_counter
from pathlib import Path
import argparse
import warnings
import pandas as pd

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from loguru import logger
from ultralytics import YOLO
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

# Local (package) imports
from cow_detectection.modeling.yolov8.predict import (
    run_inference, draw_yolov8_results, extract_bboxes
)
from cow_detectection.modeling.stgcn.predict import TSSTGInference


# ---------------------------
# Path setup (robust to /testing/)
# ---------------------------
HERE = Path(__file__).resolve().parent                       # .../cow_detectection/testing
PKG_ROOT = HERE.parent                                       # .../cow_detectection
MODELING = PKG_ROOT / "modeling"

YOLO_CKPT   = MODELING / "yolov8" / "weights" / "yolov8m.pt"
POSE_CONFIG = MODELING / "hrnet" / "config" / "hrnet_w32_ap10k_256_256.py"
POSE_CKPT   = MODELING / "hrnet" / "weights" / "hrnet_w32_ap10k.pth"
ACTION_CKPT = MODELING / "stgcn" / "weights" / "tsstg-model.pth"

RESULTS_DIR = HERE / "bench_results"        # saves visualizations (mirrors input tree)
BENCH_DIR   = HERE / "bench"                # CSV + plots

BASELINE_THR = 0.5
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------
# Helpers
# ---------------------------
def _sync_if_cuda(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def draw_action_label(frame, bbox, action_label, reserve_above_px=30,
                      color=(102, 0, 204), min_scale=0.10, max_scale=1.2,
                      padding=3, thickness=1):
    """Draw action text ABOVE the bbox; font scales with bbox size."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7 * max(min_scale, min(max_scale, 0.28 * bh / 18.0))
    (tw, th), base = cv2.getTextSize(action_label, font, scale, thickness)

    while tw + 2 * padding > bw and scale > min_scale:
        scale *= 0.9
        (tw, th), base = cv2.getTextSize(action_label, font, scale, thickness)

    ax, ay = x1, y1 - reserve_above_px
    if ay - th - base - padding < 0:
        ay = th + base + padding
    if ax + tw + 2 * padding > W:
        ax = max(0, W - tw - 2 * padding)

    bg1 = (int(ax - padding), int(ay - th - base - padding))
    bg2 = (int(ax + tw + padding), int(ay + padding - base))
    bg1 = (max(0, bg1[0]), max(0, bg1[1]))
    bg2 = (min(W - 1, bg2[0]), min(H - 1, bg2[1]))
    fill = (max(color[0] - 40, 0), max(color[1] - 40, 0), max(color[2] - 40, 0))
    cv2.rectangle(frame, bg1, bg2, fill, thickness=-1)
    cv2.putText(frame, action_label, (int(ax), int(ay - base)),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_action_labels(frame, bboxes, labels):
    for det, label in zip(bboxes, labels):
        x1, y1, x2, y2, _ = det["bbox"]
        draw_action_label(frame, (x1, y1, x2, y2), label, reserve_above_px=3)

def iter_images(root: Path):
    """Yield absolute image paths recursively under root."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            yield p


# ---------------------------
# Main Benchmark
# ---------------------------
def main(root_folder: str, device: str = "cpu", show_skeleton: bool = True,
         limit: int = 0, warmup: int = 0, save_viz: bool = True):
    root = Path(root_folder).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    # Ensure outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Load models (one-time)
    logger.info("Loading models…")
    yolo = YOLO(str(YOLO_CKPT))

    pose_model = init_pose_model(str(POSE_CONFIG), str(POSE_CKPT), device=device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info) if dataset_info else None

    tsstg = TSSTGInference(model_path=str(ACTION_CKPT), device=device)

    # Collect image list
    images = list(iter_images(root))
    images.sort()
    if limit > 0:
        images = images[:limit]
    logger.info(f"Found {len(images)} images (device={device}, show_skeleton={show_skeleton}, warmup={warmup}).")

    if not images:
        logger.warning("No images found — nothing to do.")
        return

    # Warmup frames (not recorded)
    for i, img_path in enumerate(images[:warmup], 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        _ = run_inference(yolo, frame, conf_thr=BASELINE_THR)
        bboxes = [{"bbox": [0, 0, frame.shape[1]-1, frame.shape[0]-1, 1.0]}]  # dummy if needed

    # Timings
    t_yolo, t_pose, t_action, t_total_compute = [], [], [], []
    t_yolo_per_cow, t_pose_per_cow, t_action_per_cow, t_total_per_cow = [], [], [], []

    # CSV rows
    csv_rows = []

    for idx, img_path in enumerate(images[warmup:], start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Cannot read {img_path}")
            continue

        # ---------- YOLO ----------
        _sync_if_cuda(device); t1 = perf_counter()
        detections = run_inference(yolo, frame, conf_thr=BASELINE_THR)
        _sync_if_cuda(device); t2 = perf_counter()

        cow_dets = [d for d in detections if str(d.get("cls_name","")).lower() == "cow"]
        num_cows = len(cow_dets)

        # ---------- HRNet ----------
        pose_results, bboxes = [], []
        if cow_dets:
            bboxes = extract_bboxes(cow_dets)
            _sync_if_cuda(device); t_pose_start = perf_counter()
            pose_results, _ = inference_top_down_pose_model(
                pose_model, frame, bboxes, bbox_thr=0.3, format="xyxy",
                dataset=dataset, dataset_info=dataset_info, return_heatmap=False, outputs=None
            )
            _sync_if_cuda(device); t_pose_end = perf_counter()
        else:
            t_pose_start = t_pose_end = perf_counter()

        # ---------- TS-STG ----------
        _sync_if_cuda(device); t_act_start = perf_counter()
        action_labels = []
        if pose_results:
            for animal in pose_results:
                pts = np.array(animal["keypoints"])[None, :, :]  # (1, V, C)
                action = tsstg.infer(pts, frame.shape[:2])
                action_labels.append(action)
        _sync_if_cuda(device); t_act_end = perf_counter()

        # ---------- Compute-only total ----------
        dt_yolo   = (t2 - t1) * 1000.0
        dt_pose   = (t_pose_end - t_pose_start) * 1000.0
        dt_action = (t_act_end - t_act_start) * 1000.0
        dt_total  = dt_yolo + dt_pose + dt_action

        t_yolo.append(dt_yolo)
        t_pose.append(dt_pose)
        t_action.append(dt_action)
        t_total_compute.append(dt_total)

        if num_cows > 0:
            t_yolo_per_cow.append(dt_yolo / num_cows)
            t_pose_per_cow.append(dt_pose / num_cows)
            t_action_per_cow.append(dt_action / num_cows)
            t_total_per_cow.append(dt_total / num_cows)

        # ---------- Visualization & Save (optional, not in compute total) ----------
        if save_viz:
            vis = frame.copy()
            if show_skeleton and pose_results:
                _sync_if_cuda(device)
                vis_img = vis_pose_result(
                    pose_model, frame, pose_results,
                    dataset=dataset, dataset_info=dataset_info,
                    kpt_score_thr=0.2, radius=6, thickness=3, show=False
                )
                vis = vis_img.copy()
            draw_yolov8_results(vis, cow_dets)
            if action_labels:
                draw_action_labels(vis, bboxes, action_labels)

            rel = img_path.relative_to(root)
            save_path = (RESULTS_DIR / rel).with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)

        logger.info(
            f"[{idx:03d}/{len(images)-warmup:03d}] {img_path.name:<25} "
            f"cows={num_cows:2d} | YOLO {dt_yolo:6.1f} | HRNet {dt_pose:6.1f} | TSSTG {dt_action:6.1f} | "
            f"Compute-total {dt_total:6.1f} ms | per-cow {(dt_total/num_cows) if num_cows else 0.0:.1f} ms"
        )

        # CSV row
        csv_rows.append([
            str(img_path.relative_to(root)), num_cows,
            f"{dt_yolo:.3f}", f"{dt_pose:.3f}", f"{dt_action:.3f}", f"{dt_total:.3f}"
        ])

    # ---------------------------
    # Summary stats (outside loop)
    # ---------------------------
    def _safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
    def _safe_med(arr):  return float(np.median(arr)) if arr else 0.0
    def _safe_p95(arr):  return float(np.quantile(arr, 0.95)) if arr else 0.0

    avg_yolo   = _safe_mean(t_yolo)
    avg_pose   = _safe_mean(t_pose)
    avg_act    = _safe_mean(t_action)
    avg_total  = _safe_mean(t_total_compute)

    med_total  = _safe_med(t_total_compute)
    p95_total  = _safe_p95(t_total_compute)

    avg_yolo_cow = _safe_mean(t_yolo_per_cow)
    avg_pose_cow = _safe_mean(t_pose_per_cow)
    avg_act_cow  = _safe_mean(t_action_per_cow)
    avg_total_cow= _safe_mean(t_total_per_cow)

    n_imgs = len(csv_rows)

    print("\n===== Pipeline Summary =====")
    print(f"Device:               {device.upper()}")
    print(f"Skeleton mode:        {'Shown' if show_skeleton else 'Not shown'}")
    print("---------------------------------------")
    print(f"Images processed:     {n_imgs}")
    print(f"Avg YOLOv8 time:      {avg_yolo:.1f} ms  (per cow: {avg_yolo_cow:.1f} ms)")
    print(f"Avg HRNet time:       {avg_pose:.1f} ms  (per cow: {avg_pose_cow:.1f} ms)")
    print(f"Avg TS-STG time:      {avg_act:.1f} ms   (per cow: {avg_act_cow:.1f} ms)")
    print(f"Avg compute total:    {avg_total:.1f} ms | median: {med_total:.1f} | p95: {p95_total:.1f}")
    print(f"Avg total per cow:    {avg_total_cow:.1f} ms")
    print("=======================================\n")

    # ---------------------------
    # CSV export
    # ---------------------------
    csv_path = BENCH_DIR / "pipeline_times.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_relpath", "num_cows", "yolo_ms", "hrnet_ms", "stgcn_ms", "compute_total_ms"])
        w.writerows(csv_rows)
    print(f"[OK] CSV → {csv_path}")

    # --- Simple plots to match your screenshot (hist + per-image) ---
    # We’ll read the CSV back into a DataFrame so the code mirrors your snippet
    df = pd.read_csv(csv_path)

    # Ensure numeric, and alias 'overall_ms' --> use our compute_total_ms column
    df["compute_total_ms"] = pd.to_numeric(df["compute_total_ms"], errors="coerce")
    df = df.dropna(subset=["compute_total_ms"]).reset_index(drop=True)
    df["overall_ms"] = df["compute_total_ms"]  # alias for plotting convenience

    # 1) Overall latency histogram
    plt.figure()
    df["overall_ms"].plot(kind="hist", bins=30)
    plt.xlabel("Latency (ms)")
    plt.title("Overall latency distribution")
    plt.tight_layout()
    plt.savefig(BENCH_DIR / "latency_hist.png", dpi=170)
    plt.close()

    # ---------------------------
    # Plots
    # ---------------------------
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Total time curve
    plt.figure(figsize=(7, 4))
    x = np.arange(1, len(t_total_compute) + 1)
    plt.plot(x, t_total_compute, marker="o", label="Compute total per image")
    if len(t_total_compute) >= 5:
        roll = np.convolve(t_total_compute, np.ones(5) / 5, mode="valid")
        plt.plot(np.arange(3, len(roll) + 3), roll, label="5-image rolling avg")
    plt.axhline(avg_total, color="red", linestyle="--", label=f"Avg {avg_total:.1f} ms")
    plt.title("Compute Total Time per Image")
    plt.xlabel("Image index"); plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(BENCH_DIR / "total_inference_time.png", dpi=170); plt.close()

    # 2) Average time per model (per image)
    plt.figure(figsize=(6, 4))
    models = ["YOLOv8", "HRNet", "TS-STG", "Compute total"]
    times  = [avg_yolo, avg_pose, avg_act, avg_total]
    plt.bar(models, times)
    plt.ylabel("Average time (ms)"); plt.title("Average Inference Time per Model")
    plt.tight_layout(); plt.savefig(BENCH_DIR / "avg_time_per_model.png", dpi=170); plt.close()

    # 3) Per-cow average time per model
    plt.figure(figsize=(6, 4))
    times_cow = [avg_yolo_cow, avg_pose_cow, avg_act_cow, avg_total_cow]
    plt.bar(models, times_cow)
    plt.ylabel("Average time per cow (ms)"); plt.title("Average Time per Cow")
    plt.tight_layout(); plt.savefig(BENCH_DIR / "avg_time_per_model_per_cow.png", dpi=170); plt.close()

    print(f"[OK] Plots → {BENCH_DIR}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to test image folder (recursively scanned)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    ap.add_argument("--show-skeleton", dest="show_skeleton", action="store_true", help="Overlay HRNet skeletons")
    ap.add_argument("--no-skeleton", dest="show_skeleton", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (after sort)")
    ap.add_argument("--warmup", type=int, default=0, help="Unrecorded warm-up images")
    ap.add_argument("--no-save-viz", dest="save_viz", action="store_false", help="Skip saving visualization images")
    ap.set_defaults(show_skeleton=True, save_viz=True)
    args = ap.parse_args()

    main(
        root_folder=args.root,
        device=args.device,
        show_skeleton=args.show_skeleton,
        limit=args.limit,
        warmup=args.warmup,
        save_viz=args.save_viz,
    )
