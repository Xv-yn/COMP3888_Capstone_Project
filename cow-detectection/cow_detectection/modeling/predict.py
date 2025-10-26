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
ACTION_CKPT = "stgcn/weights/tsstg-model.pth"
OUTPUT_DIR = "../results/vis_res"
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# CLI
app = typer.Typer()


def draw_action_labels(frame, bboxes, labels):
    """Draw compact red label boxes with white text above each bbox."""
    RED_BG = (102, 0, 204)  # deeper/denser red (BGR); tweak to (0,0,255) for pure red
    WHITE = (255, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.6
    THK = 2
    PAD_X, PAD_Y = 1, 1  # smaller padding for tighter box

    h_img = frame.shape[0]

    for det, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = det["bbox"]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        text = f"{label} {score:.2f}"
        (tw, th), _baseline = cv2.getTextSize(text, FONT, SCALE, THK)

        # Place label ABOVE bbox; if too close to top, place just inside
        text_x = x1
        text_y = y1 - 6
        if text_y - th - PAD_Y < 0:
            text_y = min(y1 + th + 6, h_img - 1)

        # Tight background: ignore baseline so the rect hugs the glyphs
        bg_tl = (text_x - PAD_X, text_y - th - PAD_Y)
        bg_br = (text_x + tw + PAD_X, text_y + PAD_Y)
        cv2.rectangle(frame, bg_tl, bg_br, RED_BG, thickness=-1)

        # White text on top
        cv2.putText(frame, text, (text_x, text_y), FONT, SCALE, WHITE, THK, lineType=cv2.LINE_AA)


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
    if option == 3:
        # Step 3: Action recognition
        tsstg = TSSTGInference(model_path=ACTION_CKPT, device=device)
        action_labels = []
        for animal in pose_results:
            pts = np.array(animal["keypoints"])[None, :, :]  # (1, V, C)
            action_prob = tsstg.infer(pts, frame_orig.shape[:2])
            action_labels.append(action_prob)

        draw_action_labels(frame_vis, bboxes, action_labels)

    # Step 4: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, frame_vis)
    logger.info(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    app()
