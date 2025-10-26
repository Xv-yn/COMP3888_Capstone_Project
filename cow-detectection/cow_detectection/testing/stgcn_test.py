#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GCN pipeline test (average-score acceptance), aligned with predict.py:
  YOLOv8 detection -> HRNet pose (bbox_thr=0.3) -> TS-STG action prediction

Acceptance rule (matches hernet_test/pose_test):
  average(keypoint_score) >= --thres  ==> accepted

Expected usage:
python3 stgcn_test.py \
  --data-root ./testing_set \
  --yolo-ckpt ../modeling/yolov8/weights/yolov8m.pt \
  --pose-config ../modeling/hrnet/config/hrnet_w32_ap10k_256_256.py \
  --pose-ckpt ../modeling/hrnet/weights/hrnet_w32_ap10k.pth \
  --stgcn-ckpt ../modeling/stgcn/weights/tsstg-model.pth \
  --device cuda:0 \
  --thres 0.80 --conf 0.30
"""

import argparse
from collections import Counter
import glob
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo
from ultralytics import YOLO

from cow_detectection.modeling.stgcn.predict import TSSTGInference
from cow_detectection.modeling.yolov8.predict import extract_bboxes, run_inference

# -----------------------------
# Folder → GT label mapping
# -----------------------------
FOLDER_TO_STGCN = {
    "standing": "Stand",
    "walking": "Walk",
    "lying": "Lay",
    "feeding": "Eat",
    "others": None,
    "edge": None,
}


# -----------------------------
# Helpers
# -----------------------------
def safe_imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def average_keypoint_score(pose_entry) -> float:
    """Return average keypoint score for a single MMPose result entry."""
    kpts = pose_entry.get("keypoints", None)
    if kpts is None or len(kpts) == 0:
        return 0.0
    return float(np.mean(kpts[:, 2]))


def moving_average(binary_accepts, window=50):
    out, s, q = [], 0.0, []
    for v in binary_accepts:
        q.append(1.0 if v else 0.0)
        s += q[-1]
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def compute_prf1_per_class(y_true, y_pred, classes):
    res = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        n = sum(1 for yt in y_true if yt == c)
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        res[c] = (P, R, F1, n)
    return res


def plot_visibility(scores, accepts, out_path="visibility_diagnostics.png", thres=0.8):
    """Left: cumulative acceptance; Right: histogram of average keypoint scores."""
    if not scores or not accepts:
        return
    scores = np.asarray(scores, dtype=float)
    acc = np.asarray(accepts, dtype=float)
    cum_acc = np.cumsum(acc) / np.arange(1, len(acc) + 1, dtype=float)
    final_rate = float(cum_acc[-1])

    plt.figure(figsize=(12, 4))
    # Left: cumulative acceptance
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(cum_acc) + 1), cum_acc, color="green", linewidth=2)
    ax1.axhline(final_rate, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Cumulative Acceptance Rate")
    ax1.set_xlabel("Instances Processed")
    ax1.set_ylabel("Acceptance Rate")
    ax1.set_ylim(max(0.2, np.nanmin(cum_acc) - 0.05), 1.0)

    # Right: score histogram
    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(scores, bins=25, color="#87CEEB", edgecolor="black", alpha=0.8)
    ax2.axvline(thres, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={thres:g}")
    ax2.legend(loc="upper left")
    ax2.set_title("Average-Keypoint Score Distribution")
    ax2.set_xlabel("avg_score")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, out_path="stgcn_diagnostics.png"):
    if len(y_true) == 0:
        return
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        # guard: skip any labels not in classes (shouldn't happen with our caller)
        if yt not in idx or yp not in idx:
            continue
        M[idx[yt], idx[yp]] += 1
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, interpolation="nearest")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix (counts)")
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def resolve_pred_label(tsstg: TSSTGInference, pts_xyc: np.ndarray, image_hw) -> str:
    """
    Align with predict.py semantics:
    - If tsstg.infer returns a vector/Tensor of class scores, take argmax and map via tsstg.class_names.
    - If it returns a string label, use it directly.
    """
    if pts_xyc.ndim == 2:
        pts_xyc = pts_xyc[None, :, :]
    out = tsstg.infer(pts_xyc, image_size=(image_hw[1], image_hw[0]))
    if isinstance(out, (list, tuple, np.ndarray)):
        idx = int(np.argmax(out))
        return tsstg.class_names[idx]
    if isinstance(out, torch.Tensor):
        idx = int(torch.argmax(out).item())
        return tsstg.class_names[idx]
    return str(out)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test ST-GCN pipeline accuracy (avg-score acceptance), matching predict.py behavior."
    )
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--yolo-ckpt", required=True, type=str)
    parser.add_argument("--pose-config", required=True, type=str)
    parser.add_argument("--pose-ckpt", required=True, type=str)
    parser.add_argument("--stgcn-ckpt", required=True, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--thres", default=0.80, type=float, help="ACCEPT if average keypoint score >= thres"
    )
    parser.add_argument(
        "--conf",
        default=0.30,
        type=float,
        help="Compatibility flag to match CLI; not used by avg-score gate",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="include folders 'others'/'edge' in accuracy computation",
    )
    parser.add_argument(
        "--yolo-conf", default=0.25, type=float, help="(if your run_inference uses it)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[Error] --data-root not found: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    # Models
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available. Using CPU.")
        device = "cpu"

    yolo = YOLO(args.yolo_ckpt)
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, device=device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info) if dataset_info is not None else None

    tsstg = TSSTGInference(model_path=args.stgcn_ckpt, device=device)
    class_names = list(tsstg.class_names)

    # valid GT classes are those mapped from folders (no "others"/"edge")
    valid_gt_classes = sorted(
        {v for v in FOLDER_TO_STGCN.values() if v is not None},
        key=lambda x: class_names.index(x) if x in class_names else 0,
    )

    # Data
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    samples = []
    for sub in sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    ):
        sub_dir = os.path.join(args.data_root, sub)
        for ext in image_exts:
            samples += [(p, sub) for p in glob.glob(os.path.join(sub_dir, f"*{ext}"))]
    if not samples:
        print(f"[Error] No images found under {args.data_root}", file=sys.stderr)
        sys.exit(1)

    # Eval
    y_true, y_pred = [], []
    avg_scores, accepts = [], []
    pred_dist = Counter()
    confusion_counter = Counter()

    for img_path, folder_name in tqdm(samples, desc="Evaluating", unit="img"):
        gt_label = FOLDER_TO_STGCN.get(folder_name, None)
        if gt_label is None and not args.include_unknown:
            continue

        img = safe_imread(img_path)
        H, W = img.shape[:2]

        dets = run_inference(yolo, img)
        bboxes = extract_bboxes(dets)
        if not bboxes:
            continue

        # Match predict.py: bbox_thr=0.3
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img,
            bboxes,
            bbox_thr=0.3,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        for animal in pose_results:
            avg_score = average_keypoint_score(animal)
            accept = avg_score >= args.thres
            avg_scores.append(avg_score)
            accepts.append(accept)
            if not accept:
                continue

            kpts = animal["keypoints"]
            if np.isnan(kpts[:, :2]).any():
                continue

            pred_label = resolve_pred_label(tsstg, kpts, (H, W))
            pred_dist[pred_label] += 1

            if gt_label is not None:
                y_true.append(gt_label)
                y_pred.append(pred_label)
                if gt_label != pred_label:
                    confusion_counter[(gt_label, pred_label)] += 1

    # Metrics
    n_inst = len(avg_scores)
    n_acc = sum(1 for a in accepts if a)
    acc_rate = (n_acc / n_inst) if n_inst else 0.0
    score_mean = float(np.mean(avg_scores)) if avg_scores else 0.0
    score_median = float(np.median(avg_scores)) if avg_scores else 0.0

    eval_n = len(y_true)
    top1 = (sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / eval_n) if eval_n else 0.0
    prf1 = compute_prf1_per_class(y_true, y_pred, valid_gt_classes)

    # Plots
    plot_visibility(avg_scores, accepts, out_path="visibility_diagnostics.png", thres=args.thres)

    # Build confusion-matrix classes as the union of seen labels (GT ∪ Pred)
    seen = set(y_true) | set(y_pred)
    classes_cm = [c for c in class_names if c in seen]
    if classes_cm:
        plot_confusion_matrix(y_true, y_pred, classes=classes_cm, out_path="stgcn_diagnostics.png")

    ma50 = moving_average(accepts, window=50)
    ma_tail = ma50[-1] if ma50 else 0.0

    # Pretty names
    pretty = {
        "Eat": "feeding",
        "Lay": "lying",
        "Stand": "standing",
        "Walk": "walking",
        "Run": "run",
    }

    # Summary
    print("===== TS-STG Pipeline Test Summary =====")
    print(f"Animals evaluated     : {n_inst}")
    print(f"Accepted (avg-score)  : {n_acc}")
    print(
        f"Acceptance rate       : {acc_rate:.3f}  (thres={args.thres:.2f}, conf={args.conf:.2f})"
    )
    print(f"Avg score mean        : {score_mean:.3f} | median {score_median:.3f}")
    print(f"Cumulative acceptance (last 50) ~ {ma_tail:.3f}")
    print("[plot] Saved visibility diagnostics to: visibility_diagnostics.png\n")

    print(f"[Accuracy] On {eval_n} evaluated animals (include_unknown={args.include_unknown})")
    print(f"Top-1 Accuracy        : {top1:.3f}\n")

    print("Per-class metrics (precision / recall / F1):")
    for cls in valid_gt_classes:
        P, R, F1, n = prf1[cls]
        name = pretty.get(cls, cls.lower())
        print(f"  {name:<9} n={n:4d}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}")

    if len(confusion_counter) > 0:
        print("\nTop confusions (gt → pred : count):")
        for (gt, pd), c in confusion_counter.most_common(10):
            ngt = pretty.get(gt, gt.lower())
            npd = pretty.get(pd, pd.lower())
            print(f"  {ngt:<9} → {npd:<9} : {c}")
    else:
        print("\nTop confusions (gt → pred : count):")
        print("  (none)")
    print("[plot] Saved ST-GCN diagnostics to: stgcn_diagnostics.png\n")

    if pred_dist:
        print("[Predicted label distribution]")
        for cls in class_names:
            if cls in pred_dist:
                print(f"  {pretty.get(cls, cls.lower()):<9}: {pred_dist[cls]}")
    else:
        print("[Predicted label distribution]\n  (none)")


if __name__ == "__main__":
    main()
