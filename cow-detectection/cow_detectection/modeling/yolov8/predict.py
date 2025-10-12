"""
Run YOLOv8 inference on a single image or on all images in a directory.

Usage:
    python predict.py /path/to/image_or_directory
"""

import glob
import os
import sys

from ultralytics import YOLO


def is_image_file(path):
    """
    Return True if the path looks like an image file we support.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    return path.lower().endswith(valid_exts)


def run_inference(model, image_path, output_dir="yolov8/results"):
    """
    Run YOLO inference for a single image and save results.
    """
    print(f"[INFO] Running inference on: {image_path}")
    results = model.predict(
        source=image_path, save=True, project=output_dir, name="run", exist_ok=True
    )
    print(f"[INFO] Saved results in {output_dir}/run")
    return results


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
            run_inference(model, img)

    elif os.path.isfile(input_path) and is_image_file(input_path):
        run_inference(model, input_path)

    else:
        print(f"[ERROR] Invalid path or unsupported file format: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
