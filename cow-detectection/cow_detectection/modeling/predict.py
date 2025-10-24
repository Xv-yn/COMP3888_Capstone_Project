"""
Multi-stage cow analytics: YOLOv8 detection → HRNet pose → TS-STG action recognition.

Usage:
    # 1) YOLO only (draws boxes)
    python predict.py --option 1 --image-path /path/to/image.jpg --show-skeleton/--no-show-skeleton

    # 2) YOLO + HRNet pose (boxes + skeletons)
    python predict.py --option 2 --image-path /path/to/image.jpg --device cuda --show-skeleton/--no-show-skeleton

    # 3) YOLO + HRNet + TS-STG action (boxes + skeletons + action labels)
    python predict.py --option 3 --image-path /path/to/image.jpg --device cuda --show-skeleton/--no-show-skeleton

Notes:
    - Restored visualization is written to ../results/vis_res/<image_name>.
    - Supported inputs: .jpg, .jpeg, .webp, .bmp, .png
    - --device can be "cpu" or "cuda".

# Optional flag (only for options 2 & 3)
    --no-show-skeleton     Disable drawing HRNet skeletons on animals.

"""

import argparse
import os

import cv2
from loguru import logger
from mmpose.apis import inference_top_down_pose_model  # noqa: E402
from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo  # noqa: E402
import numpy as np
import torch
import typer
from ultralytics import YOLO

from cow_detectection.modeling.stgcn.predict import TSSTGInference
# --- local imports ---
from cow_detectection.modeling.yolov8.predict import (
    draw_yolov8_results,
    extract_bboxes,
    run_inference,
)

# =========================
# Global constants
# =========================
YOLO_CKPT = "yolov8/weights/yolov8m.pt"
POSE_CONFIG = "hrnet/config/hrnet_w32_ap10k_256_256.py"
POSE_CKPT = "hrnet/weights/hrnet_w32_ap10k.pth"
ACTION_CKPT = "stgcn/results/best.pt"
OUTPUT_DIR = "../results/vis_res"
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# CLI
app = typer.Typer()
def draw_action_label(frame, bbox,action_label, reserve_above_px=30,
                      color=(102, 0, 204),
                      min_scale=0.10, max_scale=1.2,
                      padding=3, thickness=1):
    """Draw action text ABOVE the bbox; font scales with bbox."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # base scale from bbox height
    target_cap = 0.28 * bh
    scale = 0.7*max(min_scale, min(max_scale, target_cap / 18.0))
    (tw, th), base = cv2.getTextSize(action_label, font, scale, thickness)

    # shrink to fit bbox width
    while tw + 2*padding > bw and scale > min_scale:
        scale *= 0.9
        (tw, th), base = cv2.getTextSize(action_label, font, scale, thickness)

    # position above
    ax, ay = x1, y1 - reserve_above_px
    # if off top, shrink; if still off, clamp
    while (ay - th - base - padding) < 0 and scale > min_scale:
        scale *= 0.9
        (tw, th), base = cv2.getTextSize(action_label, font, scale, thickness)
    if ay - th - base - padding < 0:
        ay = th + base + padding

    # clamp horizontally
    if ax + tw + 2*padding > W: ax = max(0, W - tw - 2*padding)
    if ax < 0: ax = 0

    # background + text
    bg1 = (int(ax - padding), int(ay - th - base - padding))
    bg2 = (int(ax + tw + padding), int(ay + padding - base))
    bg1 = (max(0,bg1[0]), max(0,bg1[1]))
    bg2 = (min(W-1,bg2[0]), min(H-1,bg2[1]))
    fill = (max(color[0]-40,0), max(color[1]-40,0), max(color[2]-40,0))
    cv2.rectangle(frame, bg1, bg2, fill, thickness=-1)
    cv2.putText(frame, action_label, (int(ax), int(ay - base)),
                font, scale, (255,255,255), thickness, cv2.LINE_AA)
def draw_action_labels(frame, bboxes, labels):
    """Draw labels on frame given MMPose-style bbox dicts and label strings."""
    for det, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = det["bbox"]         
        draw_action_label(frame, (x1, y1, x2, y2), label, reserve_above_px=3)

# =========================
# Main pipeline
# =========================
@app.command()
def main(
    option: int = typer.Option(..., help="1: YOLO only, 2: YOLO + HRNet, 3: YOLO + HRNet + TSSTG"),
    image_path: str = typer.Option(..., help="Path to input image"),
    device: str = typer.Option("cpu", help="Device to run inference on: cpu or cuda"),
    show_skeleton: bool = typer.Option(
        True,
        "--show-skeleton/--no-show-skeleton",
        help="Show HRNet skeleton overlay when using HRNet or TSSTG options",
    ),
):

    import copy

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image from {image_path}")

    logger.info(f"Running detection on {image_path}")
    # Keep the original frame intact
    frame_orig = frame.copy()

    # Step 1: YOLO detection
    yolo_model = YOLO(YOLO_CKPT)
    detections = run_inference(yolo_model, frame_orig)
    frame_yolo = frame_orig.copy()
    draw_yolov8_results(frame_yolo, detections)
    # <- default output is the YOLO-only visualization
    frame_out = frame_yolo

    if option >= 2:
        # Step 2: Pose estimation
        pose_model = init_pose_model(POSE_CONFIG, POSE_CKPT, device=device)
        dataset = pose_model.cfg.data["test"]["type"]
        dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
        if dataset_info is None:
            warnings.warn("Please set `dataset_info` in the pose config.", DeprecationWarning)
            dataset_info = None
        else:
            dataset_info = DatasetInfo(dataset_info)

        bboxes = extract_bboxes(detections)
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame_orig,
            bboxes,
            bbox_thr=0.3,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        # visualize only if requested
        if show_skeleton:
            vis_img = vis_pose_result(
                pose_model,
                frame_orig,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=0.2,
                radius=8,
                thickness=4,
                show=False,
            )
            frame_vis = vis_img.copy()
        else:
            frame_vis = frame_orig.copy()

        # always draw boxes on top
        draw_yolov8_results(frame_vis, detections)
        frame_out = frame_vis  # <- now the output becomes the pose+YOLO image
        
    if option == 3:
        # Step 3: Action recognition
        tsstg = TSSTGInference(model_path=ACTION_CKPT, device=device)
        action_labels = []
        for animal in pose_results:
            pts = np.array(animal["keypoints"])[None, :, :]  # (1, V, C)
            action_prob = tsstg.infer(pts, frame_orig.shape[:2])
            score= tsstg.score(pts,frame_orig.shape[:2])
            text = f"{action_prob} {score:.2f}" 
            action_labels.append(text)
        print(action_labels)
        draw_action_labels(frame_out, bboxes, action_labels)

    # Step 4: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, frame_out)
    logger.info(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    app()
