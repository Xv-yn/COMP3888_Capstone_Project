#!/usr/bin/env python3
"""
hrnet_test.py
Run pose inference directly using MMPose (no hrnet_inference.py).

Visible-ratio mode:
  skeleton_score = (# joints with score >= joint_conf) / (total joints)
  accept if skeleton_score >= --thres

Usage:
    python hrnet_test.py --source ./testing_set --thres 0.8
"""

import argparse, os, csv
from pathlib import Path
import warnings, cv2, numpy as np, torch
from ultralytics import YOLO
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo
import matplotlib.pyplot as plt

# Auto-locate project root: cow-detectection/cow_detectection/
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent

# YOLO (detector)
DET_WEIGHTS = PROJECT_ROOT / "modeling" / "yolov8" / "weights" / "yolov8m.pt"
DET_CONF = 0.5

# HRNet (pose)
POSE_CONFIG = PROJECT_ROOT / "modeling" / "hrnet" / "config" / "hrnet_w32_ap10k_256_256.py"
POSE_CKPT   = PROJECT_ROOT / "modeling" / "hrnet" / "weights" / "hrnet_w32_ap10k.pth"

# Scoring
JOINT_CONF = 0.30     # per-joint visibility cutoff
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ------------------------------------------------------------------------------
def visible_ratio(kpts, joint_conf: float) -> float:
    if kpts is None or len(kpts) == 0:
        return 0.0
    scores = np.asarray(kpts)[:, 2].astype(float)
    return float(np.mean(scores >= joint_conf))

def main():
    ap = argparse.ArgumentParser(description="Minimal multi-cow pose tester (visible_ratio).")
    ap.add_argument("--source", type=Path, required=True, help="Folder of raw images")
    ap.add_argument("--thres",  type=float, default=0.60,
                    help="Skeleton accepted if visible_ratio ≥ this (default 0.60)")
    args = ap.parse_args()

    imgs = [p for p in args.source.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found under {args.source}")

    print(f"[prep] Found {len(imgs)} image(s) | device={DEVICE}")
    if not DET_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {DET_WEIGHTS}")
    if not (POSE_CONFIG.exists() and POSE_CKPT.exists()):
        raise FileNotFoundError(f"Missing pose config/ckpt: {POSE_CONFIG}, {POSE_CKPT}")

    # 1) Load models (fixed paths)
    det = YOLO(str(DET_WEIGHTS))

    # Find 'cow' class id automatically
    COW_CLASS_IDS = [i for i, name in det.names.items() if str(name).lower() == "cow"]
    if not COW_CLASS_IDS:
        raise ValueError(f"'cow' class not found in model.names: {det.names}")
    COW_CLASS_ID = COW_CLASS_IDS[0]    
    
    pose = init_pose_model(str(POSE_CONFIG), str(POSE_CKPT), device=DEVICE)
    dataset = pose.cfg.data["test"]["type"]
    di = pose.cfg.data["test"].get("dataset_info")
    dataset_info = DatasetInfo(di) if di is not None else (warnings.warn(
        "Please set `dataset_info` in the pose config.", DeprecationWarning) or None)

    print(f"[run] YOLO(conf={DET_CONF}) → MMPose (JOINT_CONF={JOINT_CONF})")
    total_cows, accepted = 0, 0

    vr_list = []
    pf_list = []
    OUT_DIR = TEST_DIR / "hrnet_reports"
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- CSV collectors --------
    per_cow_rows = []     # one row per cow
    per_image_rows = []   # one row per image (aggregates)

    for i, img_path in enumerate(imgs, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read {img_path}")
            continue

        # 2) Detect cows
        y = det.predict(frame, conf=DET_CONF, verbose=False)[0]
        xyxy = y.boxes.xyxy.cpu().numpy() if y.boxes is not None else np.zeros((0,4))
        cls  = y.boxes.cls.cpu().numpy().astype(int) if y.boxes is not None else np.zeros((0,), int)
        conf = y.boxes.conf.cpu().numpy() if y.boxes is not None else np.zeros((0,))
        bboxes = [
            dict(bbox=[*xyxy[k], float(conf[k])])
            for k in range(len(xyxy))
            if int(cls[k]) == COW_CLASS_ID 
        ]

        if not bboxes:
            print(f"  [{i:03d}] {img_path.name:25s}  cows=0  (skip)")
            # record an image-level row with 0 cows, so every image appears in CSV
            per_image_rows.append({
                "image": str(img_path.relative_to(args.source)),
                "num_cows": 0,
                "accepted_cows": 0,
                "accept_rate": 0.0,
                "mean_visible_ratio": 0.0
            })
            continue

        # 3) Pose for each detection
        pose_results, _ = inference_top_down_pose_model(
            pose, frame, bboxes, bbox_thr=None, format="xyxy",
            dataset=dataset, dataset_info=dataset_info, return_heatmap=False, outputs=None
        )

        # Aggregates per image
        img_vrs = []
        img_pass = 0

        # 4) Score each cow; print running acceptance
        for j, (res, bb) in enumerate(zip(pose_results, bboxes), 1):
            kpts = res["keypoints"]
            vr = visible_ratio(kpts, JOINT_CONF)
            passed = vr >= args.thres
            total_cows += 1
            accepted += int(passed)
            rate_so_far = accepted / total_cows
            vis = int(np.sum(np.asarray(kpts)[:, 2] >= JOINT_CONF))
            total_j = len(kpts)
            det_conf = float(bb["bbox"][4]) if isinstance(bb, dict) and "bbox" in bb and len(bb["bbox"]) >= 5 else None
            mark = "✓" if passed else "✗"
            vr_list.append(vr)
            pf_list.append(int(passed))
            img_vrs.append(vr)
            img_pass += int(passed)

            # --- per-cow CSV row ---
            per_cow_rows.append({
                "image": str(img_path.relative_to(args.source)),
                "cow_id": j,
                "det_conf": det_conf,
                "visible_ratio": round(vr, 4),
                "visible_joints": vis,
                "total_joints": total_j,
                "passed": int(passed),
                "threshold": args.thres,
                "joint_conf": JOINT_CONF,
                "device": DEVICE,
            })

            print(f"  [{i:03d}] {img_path.name:25s}  cow#{j:02d}  vratio={vr:5.2f} ({vis}/{total_j})  {mark}  "
                  f"| accepted so far: {accepted}/{total_cows} = {rate_so_far:.3f}")

        # --- per-image CSV row (aggregates) ---
        mean_vr = float(np.mean(img_vrs)) if img_vrs else 0.0
        per_image_rows.append({
            "image": str(img_path.relative_to(args.source)),
            "num_cows": len(pose_results),
            "accepted_cows": img_pass,
            "accept_rate": (img_pass / len(pose_results)) if pose_results else 0.0,
            "mean_visible_ratio": mean_vr
        })

    # 5) Summary
    rate = accepted / total_cows if total_cows else 0.0
    print("\n===== Pose Test Summary =====")
    print(f"Cows evaluated      : {total_cows}")
    print(f"Accepted skeletons  : {accepted}")
    print(f"Acceptance rate     : {rate:.3f}")
    print(f"Mode                : visible_ratio (fixed)")
    print(f"Skeleton threshold  : {args.thres:.2f}")
    print(f"Joint conf (fixed)  : {JOINT_CONF:.2f}")
    print(f"Weights: YOLO={DET_WEIGHTS.name}, HRNet={POSE_CKPT.name}")

    # ------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------
    per_cow_csv = OUT_DIR / "pose_visible_ratio_per_cow.csv"
    per_image_csv = OUT_DIR / "pose_visible_ratio_per_image.csv"

    # per-cow
    if per_cow_rows:
        cow_header = [
            "image", "cow_id", "det_conf",
            "visible_ratio", "visible_joints", "total_joints",
            "passed", "threshold", "joint_conf", "device"
        ]
        with per_cow_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cow_header)
            w.writeheader()
            for r in per_cow_rows:
                w.writerow(r)
        print(f"[save] Per-cow CSV → {per_cow_csv}")
    else:
        print("[save] No cow rows to write.")

    # per-image
    if per_image_rows:
        img_header = ["image", "num_cows", "accepted_cows", "accept_rate", "mean_visible_ratio"]
        with per_image_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=img_header)
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)
        print(f"[save] Per-image CSV → {per_image_csv}")
    else:
        print("[save] No image rows to write.")

    # ------------------------------------------------------------
    # Visualization summary
    # ------------------------------------------------------------
    if vr_list:
        print(f"[plot] Generating visible-ratio summary plots...")

        # 1) Histogram of visible_ratio
        plt.figure()
        plt.hist(vr_list, bins=20, edgecolor='black', color='skyblue')
        plt.axvline(args.thres, linestyle='--', color='red', linewidth=1.5, label=f"Threshold={args.thres}")
        plt.title("Visible-Ratio Distribution")
        plt.xlabel("visible_ratio")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "hist_visible_ratio.png", dpi=150)
        plt.close()

        # 2) Cumulative acceptance rate
        cum_acc = []
        acc, seen = 0, 0
        for pf in pf_list:
            seen += 1
            acc += pf
            cum_acc.append(acc / seen)
        plt.figure()
        plt.plot(range(1, len(cum_acc)+1), cum_acc, linewidth=2, color='green')
        plt.axhline(sum(pf_list)/len(pf_list), linestyle='--', color='red', linewidth=1.2,
                    label=f"Final Rate={sum(pf_list)/len(pf_list):.3f}")
        plt.title("Cumulative Acceptance Rate")
        plt.xlabel("Instances Processed")
        plt.ylabel("Acceptance Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cumulative_acceptance.png", dpi=150)
        plt.close()

        print(f"[save] Plots saved in: {OUT_DIR}")
    else:
        print("[plot] No skeletons processed; skipping plots.")


if __name__ == "__main__":
    main()
