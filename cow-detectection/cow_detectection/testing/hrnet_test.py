#!/usr/bin/env python3
"""
pose_test.py
Run pose inference directly using MMPose (no hrnet_inference.py).

Usage:
    python pose_test.py --source ./yolo_results --thres 0.8
"""

import argparse
from pathlib import Path
import shutil
import cv2
import warnings
import numpy as np
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------------------------------------------------------------
# utility
# ---------------------------------------------------------------
def copy_images(src: Path, dst: Path):
    """Copy all images from source into pose model input folder."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images with {VALID_EXTS} under: {src}")
    for p in imgs:
        shutil.copy2(p, dst / p.name)
    return len(imgs)

def visible_ratio_score(keypoints):
    """Compute ratio of visible joints."""
    if len(keypoints) == 0:
        return 0.0
    visible = sum(1 for (x, y, s) in keypoints if s > 0)
    return visible / len(keypoints)

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run pose model test on YOLO results.")
    ap.add_argument("--source", type=Path, default=Path("./yolo_results"),
                    help="Folder containing YOLO detection/crop images")
    ap.add_argument("--thres", type=float, default=0.8,
                    help="Skeleton passes if visible_ratio >= this threshold")
    ap.add_argument("--device", default="cuda:0", help="Device to run inference")
    ap.add_argument("--pose-config", type=Path, required=True,
                    help="Path to pose model config file")
    ap.add_argument("--pose-ckpt", type=Path, required=True,
                    help="Path to pose model checkpoint file")
    args = ap.parse_args()

    imgs = [p for p in args.source.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found under {args.source}")

    print(f"[prep] Found {len(imgs)} image(s) under {args.source}")

    # ---------------------------------------------------------------
    # initialize model
    # ---------------------------------------------------------------
    pose_model = init_pose_model(str(args.pose_config), str(args.pose_ckpt), device=args.device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
            warnings.warn("Please set `dataset_info` in the pose config.", DeprecationWarning)
            dataset_info = None
    else:
        dataset_info = DatasetInfo(dataset_info)

    # ---------------------------------------------------------------
    # run inference + visible ratio evaluation
    # ---------------------------------------------------------------
    print(f"[run] Running pose inference on {len(imgs)} images...")
    accepted, total = 0, 0
    results_summary = []

    for i, img_path in enumerate(imgs, 1):
        frame = cv2.imread(str(img_path))
        height, width = frame.shape[:2]
        person = [dict(bbox=[0, 0, width, height, 1.0])]
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame,
            person,
            bbox_thr=0.3,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        if not pose_results:
            vis_ratio = 0.0
        else:
            keypoints = pose_results[0]["keypoints"]  # shape [n, 3]
            vis_ratio = visible_ratio_score(keypoints)

        passed = vis_ratio >= args.thres
        total += 1
        accepted += int(passed)
        status = "✓" if passed else "✗"
        print(f"  [{i:03d}] {img_path.name:25s}  visible_ratio={vis_ratio:5.2f}  {status}")
        results_summary.append(dict(image=str(img_path), visible_ratio=vis_ratio, passed=passed))

    # ---------------------------------------------------------------
    # summary
    # ---------------------------------------------------------------
    rate = accepted / total if total else 0.0
    print("\n===== Pose Test Summary =====")
    print(f"Images tested       : {total}")
    print(f"Accepted skeletons  : {accepted}")
    print(f"Acceptance rate     : {rate:.3f}")
    print(f"Mode                : visible_ratio")
    print(f"Skeleton threshold  : {args.thres:.2f}")

if __name__ == "__main__":
    main()
