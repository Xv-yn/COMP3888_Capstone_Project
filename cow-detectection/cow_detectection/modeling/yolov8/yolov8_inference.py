"""
detect_yolov8.py
----------------
YOLOv8 video inference for cattle detection.
Outputs:
    - detections.json (bounding boxes per frame)
    - cropped cow images for HRNet
    - annotated video (optional visualization)
"""

import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================================
# Configuration
# ==========================================================
VIDEO_PATH = "data/videos/sample.mp4"
WEIGHTS = "weights/yolov8m.pt"
SAVE_DIR = Path("results")
CROP_DIR = Path("data/crops")
CONF_THRES = 0.3
TARGET_CLASS = "cow"  # adjust to match your dataset label names
DRAW_BOX = True  # set False if you don't need visualization


# ==========================================================
# Setup
# ==========================================================
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLOv8 model
model = YOLO(WEIGHTS)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
out_path = SAVE_DIR / "annotated.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {VIDEO_PATH}")
print(f"Saving results to: {SAVE_DIR}\n")

results_json = []

# ==========================================================
# Frame-by-frame detection
# ==========================================================
for frame_idx in tqdm(range(frame_count), desc="Detecting"):
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(frame, conf=CONF_THRES, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            if label != TARGET_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Append to detections list
            results_json.append(
                {
                    "frame_id": frame_idx,
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                }
            )

            # Save cropped region
            crop = frame[y1:y2, x1:x2]
            crop_path = CROP_DIR / f"frame{frame_idx:06d}_{label}.jpg"
            cv2.imwrite(str(crop_path), crop)

            # Draw box on frame
            if DRAW_BOX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # Write annotated frame
    if DRAW_BOX:
        out.write(frame)

cap.release()
out.release()

# ==========================================================
# Save detections
# ==========================================================
json_path = SAVE_DIR / "detections.json"
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=4)

print("\nInference complete!")
print(f"Detections saved to: {json_path}")
if DRAW_BOX:
    print(f"Annotated video saved to: {out_path}")
print(f"Crops ready for HRNet: {CROP_DIR}")
