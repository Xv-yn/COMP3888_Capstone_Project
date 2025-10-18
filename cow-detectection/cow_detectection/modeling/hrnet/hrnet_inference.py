"""
extract_keypoints.py
--------------------
Stage 2 of the pipeline: Pose Estimation with pretrained HRNet (AP-10K).

What this script does:
    1) Loads cropped animal images (e.g., from a YOLO stage) from data/input/
    2) Runs HRNet (Top-Down) to produce per-joint heatmaps
    3) Converts heatmaps -> (x, y, score) keypoints in original image coords
    4) Optionally draws skeleton overlays to results/vis/
    5) Saves all keypoints to results/keypoints.pkl for downstream ST-GCN

Usage:
    python extract_keypoints.py

Notes:
    - Expects HRNet weights at weights/hrnet_w32_ap10k.pth
    - Writes:
        • results/keypoints.pkl (list of dicts)
        • results/vis/<image>.jpg (visualization)
    - Adjust CONF_THRES and IMAGE_SIZE to match your model/config.
"""

import os
from pathlib import Path
import pickle

from config.ap10k import dataset_info
import cv2
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
import numpy as np
import torch
from torchvision import transforms

# =====================================================
# Configuration
# =====================================================
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "input"
WEIGHTS_PATH = ROOT / "weights" / "hrnet_w32_ap10k.pth"
OUTPUT_DIR = ROOT / "results"
VIS_DIR = OUTPUT_DIR / "vis"
OUTPUT_PKL = OUTPUT_DIR / "keypoints.pkl"

IMAGE_SIZE = (256, 256)
CONF_THRES = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
print(f"Running on device: {DEVICE}")

# =====================================================
# 1. Build HRNet model
# =====================================================
cfg_dict = dict(
    model=dict(
        type="TopDown",
        backbone=dict(
            type="HRNet",
            in_channels=3,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block="BOTTLENECK",
                    num_blocks=(4,),
                    num_channels=(64,),
                ),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block="BASIC",
                    num_blocks=(4, 4),
                    num_channels=(32, 64),
                ),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block="BASIC",
                    num_blocks=(4, 4, 4),
                    num_channels=(32, 64, 128),
                ),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block="BASIC",
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(32, 64, 128, 256),
                ),
            ),
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=32,
            out_channels=channel_cfg["num_output_channels"],
            num_deconv_layers=0,
            extra=dict(
                final_conv_kernel=1,
            ),
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True, post_process="default", shift_heatmap=True, modulate_kernel=11
        ),
    )
)
cfg = Config(cfg_dict)
model = build_posenet(cfg.model)
_ = load_checkpoint(model, str(WEIGHTS_PATH), map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

# =====================================================
# 2. Preprocessing
# =====================================================
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    return img, tensor


# =====================================================
# 3. Inference loop
# =====================================================
results = []
for img_file in sorted(INPUT_DIR.glob("*.jpg")):
    orig_img, tensor = preprocess_image(img_file)
    with torch.no_grad():
        output = model(tensor)
        heatmaps = output.detach().cpu().numpy()[0]  # (num_joints, H, W)

    num_joints = heatmaps.shape[0]
    hmap_h, hmap_w = heatmaps.shape[1:]
    keypoints = []
    for j in range(num_joints):
        hmap = heatmaps[j]
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        conf = hmap[y, x]
        if conf < CONF_THRES:
            keypoints.append({"x": None, "y": None, "score": float(conf)})
            continue
        # Scale back to original image size
        x = (x / hmap_w) * orig_img.shape[1]
        y = (y / hmap_h) * orig_img.shape[0]
        keypoints.append({"x": float(x), "y": float(y), "score": float(conf)})

    results.append({"image": img_file.name, "skeleton": keypoints})

    # Optional visualization
    vis_img = orig_img.copy()
    for link in dataset_info["skeleton_info"].values():
        a, b = link["link"]
        a_id = next(i for i, v in dataset_info["keypoint_info"].items() if v["name"] == a)
        b_id = next(i for i, v in dataset_info["keypoint_info"].items() if v["name"] == b)
        pa, pb = keypoints[a_id], keypoints[b_id]
        if None not in (pa["x"], pb["x"]):
            cv2.line(
                vis_img, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), (0, 255, 0), 2
            )
    for kp in keypoints:
        if kp["x"] is not None:
            cv2.circle(vis_img, (int(kp["x"]), int(kp["y"])), 3, (0, 0, 255), -1)
    cv2.imwrite(str(VIS_DIR / img_file.name), vis_img)

# =====================================================
# 4. Save results to .pkl only
# =====================================================
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(results, f)

print(f"Done! Saved {len(results)} skeletons to {OUTPUT_PKL}")
