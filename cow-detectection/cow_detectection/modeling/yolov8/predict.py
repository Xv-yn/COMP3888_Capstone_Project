"""
Run YOLOv8 inference on a single image or on all images in a directory.

Usage:
    python predict.py /path/to/image_or_directory
"""

import glob
import os
import sys
import numpy as np
from ultralytics import YOLO
import cv2

def is_image_file(path):
    """
    Return True if the path looks like an image file we support.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    return path.lower().endswith(valid_exts)


def run_inference(model, frame, output_dir=None):
    """
    Run YOLO inference for a single image.
    - If `output_dir` is a non-empty string, predictions are saved under that folder.
    - If `output_dir` is empty/None, images are NOT saved.
    """
    h0, w0 = frame.shape[:2]  # original frame size

    # YOLO predict on the frame
    kwargs = dict(source=frame, exist_ok=True, save=False)
    if output_dir:
        kwargs.update(save=True, project=output_dir, name="run")

    results = model.predict(**kwargs)
    result = results[0]

    # YOLOv8 provides boxes relative to the resized input; need scaling
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Get original image size used by YOLO
    orig_h, orig_w = result.orig_shape  # no [0], just unpack the tuple
    scale_x = w0 / orig_w
    scale_y = h0 / orig_h

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        # scale boxes to original frame
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        cls_id = int(classes[i])
        score = scores[i]

        detections.append({
            "bbox": [x1, y1, x2, y2, score],
            "cls_id": cls_id,
            "cls_name": model.names[cls_id],
            "score": score
        })

    if output_dir:
        print(f"[INFO] Saved results in {output_dir}/run")
    else:
        print("[INFO] Results not saved (output_dir is empty).")

    return detections


def draw_yolov8_results(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2, score = det["bbox"]
        label = f"{det['cls_name']} {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 0, 204), 2)

def extract_bboxes(detections, conf_threshold=0.5):
    """Convert YOLO detections to MMPose-compatible format."""
    ret_bbox = []
    for det in detections:
        x1, y1, x2, y2, score = det["bbox"]
        if score >= conf_threshold:
            ret_bbox.append({
                "bbox": [x1, y1, x2, y2, score],
                "cls_id": det["cls_id"],
                "cls_name": det["cls_name"],
                "score": score
            })
    return ret_bbox



def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path/to/image/or/folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    model_path = os.path.join("weights", "yolov8m.pt")

    # Load model
    model = YOLO(model_path)

    # If path is directory → get all image files
    if os.path.isdir(input_path):
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))

        if not image_files:
            print("[ERROR] No image files found in directory.")
            sys.exit(1)

        for img in image_files:
            run_inference(model, img, output_dir="results")

    elif os.path.isfile(input_path) and is_image_file(input_path):
        run_inference(model, input_path)

    else:
        print(f"[ERROR] Invalid path or unsupported file format: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
