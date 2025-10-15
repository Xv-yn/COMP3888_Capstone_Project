import argparse
import os
import cv2
import numpy as np
import torch
import typer
from loguru import logger

from ultralytics import YOLO
from mmpose.apis import inference_topdown, init_model, visualize
# --- local imports ---
from cow_detectection.modeling.yolov8.predict import run_inference, draw_yolov8_results
from cow_detectection.modeling.stgcn.predict import TSSTGInferenc

# =========================
# Global constants
# =========================
YOLO_CKPT = "cow_detectection/modeling/yolov8/weights/yolov8m.pt"
POSE_CONFIG = "cow_detectection/modeling/hrnet/config/hrnet_w32_ap10k_256_256.py"
POSE_CKPT = "cow_detectection/modeling/hrnet/weights/hrnet_w32_ap10k.pth"
ACTION_CKPT = "cow_detectection/modeling/stgcn/weights/tsstg-model.pth"
OUTPUT_DIR = "../results/vis_res"
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# CLI
app = typer.Typer()

def extract_bboxes(detections):
    """Convert YOLOv8 detections to bounding boxes for HRNet."""
    bboxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # filter weak detections
            bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes)

def draw_action_labels(frame, bboxes, labels):
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# =========================
# Main pipeline
# =========================
@app.command()
def main(
    option: int = typer.Option(..., help="1: YOLO only, 2: YOLO + HRNet, 3: YOLO + HRNet + TSSTG"),
    image_path: str = typer.Option(..., help="Path to input image"),
    device: str = typer.Option("cpu", help="Device to run inference on: cpu or cuda"),
):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image from {image_path}")

    logger.info(f"Running detection on {image_path}")

    # Step 1: YOLO detection
    yolo_model = YOLO(YOLO_CKPT)
    detections = run_inference(yolo_model, frame)
    draw_yolov8_results(frame, detections)

    if option >= 2:
        # Step 2: Pose estimation
        pose_model = init_model(POSE_CONFIG, POSE_CKPT, device=device)
        bboxes = extract_bboxes(detections)
        pose_results, _ = inference_topdown(
            model=pose_model,
            img=frame,
            bboxes=bboxes,
            bbox_format='xyxy'
        )


        # visualize keypoints using mmpose API
        keypoints_list = [np.array(a['keypoints']) for a in pose_results]
        keypoints_array = np.array(keypoints_list)
        frame = visualize(
            img=frame,
            keypoints=keypoints_array,
            keypoint_score=keypoints_array[..., 2],
            metainfo=None,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=False,
            kpt_thr=0.3
        )


    if option == 3:
        # Step 3: Action recognition
        tsstg = TSSTGInference(model_path=ACTION_CKPT, device=device)
        action_labels = []
        for animal in pose_results:
            pts = np.array(animal["keypoints"])[None, :, :]  # (1, V, C)
            action_prob = tsstg.predict(pts, frame.shape[:2])
            label = tsstg.class_names[np.argmax(action_prob)]
            action_labels.append(label)

        draw_action_labels(frame, bboxes, action_labels)

    # Step 4: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, frame)
    logger.info(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    app()
