#!/usr/bin/env python3
"""
Multi-stage Livestock Analytics Benchmark
YOLOv8 → HRNet → TS-STG with timing and performance plots.

Usage:
    python pipeline_test_all.py --root ../testing/testing_set --device cpu --show-skeleton
"""

import os, time, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from loguru import logger
from ultralytics import YOLO
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo


from cow_detectection.modeling.yolov8.predict import run_inference, draw_yolov8_results, extract_bboxes
from cow_detectection.modeling.stgcn.predict import TSSTGInference

YOLO_CKPT = "yolov8/weights/yolov8m.pt"
POSE_CONFIG = "hrnet/config/hrnet_w32_ap10k_256_256.py"
POSE_CKPT = "hrnet/weights/hrnet_w32_ap10k.pth"
ACTION_CKPT = "stgcn/weights/tsstg-model.pth"

OUTPUT_DIR = "../results"
OUTPUT_PLOT = "../results"
BASELINE_THR = 0.5
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# =========================
# Helper functions
# =========================
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

# =========================
# Main
# =========================
def main(root_folder, device="cpu", show_skeleton=True):
    # --- Load models ---
    logger.info("Loading models...")
    yolo = YOLO(YOLO_CKPT)
    pose_model = init_pose_model(POSE_CONFIG, POSE_CKPT, device=device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info) if dataset_info else None
    tsstg = TSSTGInference(model_path=ACTION_CKPT, device=device)

    # --- Paths ---
    root = Path(root_folder)
    out_dir, out_plot = Path(OUTPUT_DIR), Path(OUTPUT_PLOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)

    MAX_IMAGES = 100
    image_paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    if len(image_paths) > MAX_IMAGES:
        image_paths = image_paths[:MAX_IMAGES]
        print(f"[INFO] Limiting to first {MAX_IMAGES} images for testing.")
    if not image_paths:
        print(f"No images found in {root}")
        return
    n_imgs = len(image_paths)
    logger.info(f"Processing {n_imgs} images on device={device}")

    t_yolo, t_pose, t_action, t_total = [], [], [], []
    t_yolo_per_cow, t_pose_per_cow, t_action_per_cow, t_total_per_cow = [], [], [], []
    all_actions = []

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Cannot read {img_path}")
            continue

        t0 = time.time()

        # ---------- YOLO ----------
        t1 = time.time()
        detections = run_inference(yolo, frame, conf_thr=BASELINE_THR)
        cow_dets = [d for d in detections if d["cls_name"].lower() == "cow"]
        num_cows = len(cow_dets)
        t2 = time.time()

        # ---------- HRNet ----------
        pose_results, bboxes = [], []
        if cow_dets:
            bboxes = extract_bboxes(cow_dets)
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                frame,
                bboxes,
                bbox_thr=0.3,
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None,
            )
        t3 = time.time()

        # ---------- TS-STG ----------
        action_labels = []
        if pose_results:
            for animal in pose_results:
                pts = np.array(animal["keypoints"])[None, :, :]
                action = tsstg.infer(pts, frame.shape[:2])
                action_labels.append(action)
                all_actions.append(action)
        t4 = time.time()

        # ---------- Record times ----------
        dt_yolo   = (t2 - t1) * 1000
        dt_pose   = (t3 - t2) * 1000
        dt_action = (t4 - t3) * 1000
        dt_total  = (t4 - t0) * 1000

        t_yolo.append(dt_yolo)
        t_pose.append(dt_pose)
        t_action.append(dt_action)
        t_total.append(dt_total)

        # avoid div-by-zero for per-cow avg
        if num_cows > 0:
            t_yolo_per_cow.append(dt_yolo / num_cows)
            t_pose_per_cow.append(dt_pose / num_cows)
            t_action_per_cow.append(dt_action / num_cows)
            t_total_per_cow.append(dt_total / num_cows)

        # ---------- Visualization ----------
        vis = frame.copy()
        if show_skeleton and pose_results:
            vis_img = vis_pose_result(
                pose_model,
                frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=0.2,
                radius=6,
                thickness=3,
                show=False,
            )
            vis = vis_img.copy()

        draw_yolov8_results(vis, cow_dets)
        if action_labels:
            draw_action_labels(vis, bboxes, action_labels)

        relative_path = img_path.relative_to(root)
        save_path = out_dir / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)

        logger.info(f"[{idx:03d}/{n_imgs:03d}] {img_path.name:<25} "
                    f"cows={num_cows:2d} | YOLO {dt_yolo:6.1f} | "
                    f"HRNet {dt_pose:6.1f} | TSSTG {dt_action:6.1f} | "
                    f"Total {dt_total:6.1f} ms | per-cow {dt_total/num_cows if num_cows>0 else 0:.1f} ms")

    # ===================================================
    # Summary stats
    # ===================================================
        avg_yolo, avg_pose, avg_act, avg_total = map(np.mean, [t_yolo, t_pose, t_action, t_total])
    avg_yolo_cow = np.mean(t_yolo_per_cow) if t_yolo_per_cow else 0
    avg_pose_cow = np.mean(t_pose_per_cow) if t_pose_per_cow else 0
    avg_act_cow  = np.mean(t_action_per_cow) if t_action_per_cow else 0
    avg_total_cow= np.mean(t_total_per_cow) if t_total_per_cow else 0

    print("\n===== Pipeline Summary (First 100 Images) =====")
    print(f"Images processed:     {n_imgs}")
    print(f"Avg YOLOv8 time:      {avg_yolo:.1f} ms  (per cow: {avg_yolo_cow:.1f} ms)")
    print(f"Avg HRNet time:       {avg_pose:.1f} ms  (per cow: {avg_pose_cow:.1f} ms)")
    print(f"Avg TS-STG time:      {avg_act:.1f} ms   (per cow: {avg_act_cow:.1f} ms)")
    print(f"Avg total time:       {avg_total:.1f} ms (per cow: {avg_total_cow:.1f} ms)")


    # ===================================================
    # Plots
    # ===================================================
    out_plot.mkdir(parents=True, exist_ok=True)

    # 1. Total time curve
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(t_total)+1), t_total, marker="o", label="Total time per image")
    if len(t_total) >= 5:
        roll = np.convolve(t_total, np.ones(5)/5, mode="valid")
        plt.plot(range(3, len(roll)+3), roll, color="orange", label="5-image rolling avg")
    plt.axhline(avg_total, color="red", linestyle="--", label=f"Avg {avg_total:.1f} ms")
    plt.title("Total Inference Time per Image (first 100)")
    plt.xlabel("Image index")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot / "total_inference_time.png")

    # 2. Average time per model (per image)
    plt.figure(figsize=(6,4))
    models = ["YOLOv8", "HRNet", "TS-STG", "Total"]
    times = [avg_yolo, avg_pose, avg_act, avg_total]
    plt.bar(models, times, color=["#80b1d3", "#b3de69", "#fb8072", "#bc80bd"])
    plt.ylabel("Average time (ms)")
    plt.title("Average Inference Time per Model (per image)")
    plt.tight_layout()
    plt.savefig(out_plot / "avg_time_per_model.png")

    # 3️. Per-cow average time per model
    plt.figure(figsize=(6,4))
    models = ["YOLOv8", "HRNet", "TS-STG", "Total"]
    times_cow = [avg_yolo_cow, avg_pose_cow, avg_act_cow, avg_total_cow]
    plt.bar(models, times_cow, color=["#80b1d3", "#b3de69", "#fb8072", "#bc80bd"])
    plt.ylabel("Average time per cow (ms)")
    plt.title("Average Inference Time per Model (per cow)")
    plt.tight_layout()
    plt.savefig(out_plot / "avg_time_per_model_per_cow.png")


    print(f"[INFO] Saved plots → {out_plot}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to test image folder")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--show-skeleton", action="store_true", help="Overlay HRNet skeletons")
    args = ap.parse_args()
    main(args.root, args.device, args.show_skeleton)
