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


def draw_yolov8_results(frame, detections,
                        box_color=(102, 0, 204),
                        min_scale=0.10, max_scale=1.2,
                        padding=3, thickness=1):
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2, score = det["bbox"]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{det['cls_name']} {score:.2f}"

        # draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # adaptive font scale from bbox size
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        target_cap = 0.28 * bh
        scale = 0.7*max(min_scale, min(max_scale, target_cap / 18.0))

        (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
        while tw + 2*padding > bw and scale > min_scale:
            scale *= 0.7
            (tw, th), base = cv2.getTextSize(label, font, scale, thickness)

        # place label inside-top (fallback to inside-bottom)
        tx = x1
        ty = y1 + th + padding + 1
        if ty + base > y2:
            ty = max(y1 + th + padding, y2 - base - 1)

        # clamp horizontally to image
        if tx + tw + 2*padding > W: tx = max(0, W - tw - 2*padding)
        if tx < 0: tx = 0

        # label background
        bg1 = (int(tx - padding), int(ty - th - base - padding))
        bg2 = (int(tx + tw + padding), int(ty + padding - base))
        bg1 = (max(0,bg1[0]), max(0,bg1[1]))
        bg2 = (min(W-1,bg2[0]), min(H-1,bg2[1]))
        fill = (max(box_color[0]-40,0), max(box_color[1]-40,0), max(box_color[2]-40,0))
        cv2.rectangle(frame, bg1, bg2, fill, thickness=-1)

        # text
        cv2.putText(frame, label, (int(tx), int(ty - base)),
                    font, scale, (255,255,255), thickness, cv2.LINE_AA)
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
