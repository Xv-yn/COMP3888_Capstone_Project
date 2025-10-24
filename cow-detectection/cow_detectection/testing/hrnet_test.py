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
def average_score_passes_threshold(keypoints, threshold):
    """Return True if average keypoint score >= threshold."""
    if len(keypoints) == 0:
        return False
    scores = [float(s) for (_, _, s) in keypoints]
    avg_score = sum(scores) / len(scores)
    return avg_score >= threshold

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run pose model test on YOLO results.")
    ap.add_argument("--source", type=Path, default=Path("./yolo_results"),
                    help="Folder containing YOLO detection/crop images")
    ap.add_argument("--thres", type=float, default=0.8,
                    help="Skeleton passes if average score >= this threshold")
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

    # Initialize pose model
    pose_model = init_pose_model(str(args.pose_config), str(args.pose_ckpt), device=args.device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("Please set `dataset_info` in the pose config.", DeprecationWarning)
        dataset_info = None
    else:
        dataset_info = DatasetInfo(dataset_info)

    print(f"[run] Running pose inference on {len(imgs)} images...")
    accepted, total = 0, 0
    results_summary = []

    for i, img_path in enumerate(imgs, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        height, width = frame.shape[:2]
        person = [dict(bbox=[0, 0, width, height, 1.0])]  # full image as bbox

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

        keypoint_scores = []
        avg_score = 0.0
        if pose_results:
            keypoints = pose_results[0]["keypoints"]
            keypoint_scores = [round(float(k[2]), 3) for k in keypoints]
            avg_score = round(np.mean([k[2] for k in keypoints]), 3)

        passed = average_score_passes_threshold(pose_results[0]["keypoints"] if pose_results else [], args.thres)
        total += 1
        accepted += int(passed)
        status = "✓" if passed else "✗"

        print(f"  [{i:03d}] {img_path.name:25s}  avg_score={avg_score:5.2f}  {status}")
        if keypoint_scores:
            print(f"         Keypoint scores: {keypoint_scores}")

        results_summary.append(dict(
            image=str(img_path),
            avg_score=avg_score,
            passed=passed,
            keypoint_scores=keypoint_scores
        ))

    # Summary
    rate = accepted / total if total else 0.0
    print("\n===== Pose Test Summary =====")
    print(f"Images tested       : {total}")
    print(f"Accepted skeletons  : {accepted}")
    print(f"Acceptance rate     : {rate:.3f}")
    print(f"Mode                : average_score")
    print(f"Skeleton threshold  : {args.thres:.2f}")

if __name__ == "__main__":
    main()