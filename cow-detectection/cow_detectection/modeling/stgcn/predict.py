"""
TS-STG (Two-Stream Spatial-Temporal Graph) action inference for cow keypoints.

This module wraps a pretrained TS-STG model and provides a simple `.infer(...)`
API that takes a single-frame set of keypoints and returns an action label.

Pipeline inside `infer`:
    1) Repeat the single-frame keypoints to a short clip (T=60) expected by the model.
    2) Normalize keypoints by image size to [0, 1], then per-clip scale to [-1, 1].
    3) Append a pelvis/torso midpoint keypoint to stabilize geometry.
    4) Build two streams:
        - Pose stream:  (C=coords, T=frames, V=nodes)
        - Motion stream: frame-to-frame differences (velocity)
    5) Run the two-stream GCN and return the top-1 action name.

Usage:
    tsstg = TSSTGInference(model_path="stgcn/weights/tsstg-model.pth", device="cuda")
    # pts shape: (1, V, C) or (V, C) with C >= 2 (x, y[, score])
    action = tsstg.infer(pts, image_size=(W, H))
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import torch.nn.functional as F

from cow_detectection.modeling.base import BaseInference
from cow_detectection.modeling.stgcn.model import TwoStreamSpatialTemporalGraph

app = typer.Typer()


def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


class TSSTGInference(BaseInference):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.graph_args = {"strategy": "spatial"}
        self.class_names = ["Stand", "Walk", "Run", "Lay", "Eat"]
        self.num_class = len(self.class_names)

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)

        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def infer(self, pts: np.ndarray, image_size: tuple = (1920, 1080)) -> str:
        pts = np.repeat(pts, 60, axis=0)
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], *image_size)
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        midpoint = ((pts[:, 3, :] + pts[:, 4, :]) / 2).reshape(pts.shape[0], 1, -1)
        pts = np.concatenate((pts, midpoint), axis=1)

        pts_tensor = torch.tensor(pts, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        motion_tensor = pts_tensor[:, :2, 1:, :] - pts_tensor[:, :2, :-1, :]

        pts_tensor = pts_tensor.to(self.device)
        motion_tensor = motion_tensor.to(self.device)

        with torch.no_grad():
            output = self.model((pts_tensor, motion_tensor))

        pred_index = torch.argmax(output, dim=1).item()
        return self.class_names[pred_index]
    def score(self, pts: np.ndarray, image_size: tuple = (1920, 1080)) -> float:
        # build clip + streams (same as your infer)
        pts = np.repeat(pts, 60, axis=0)
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], *image_size)
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        midpoint = ((pts[:, 3, :] + pts[:, 4, :]) / 2).reshape(pts.shape[0], 1, -1)
        pts = np.concatenate((pts, midpoint), axis=1)

        pts_tensor = torch.tensor(pts, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        motion_tensor = pts_tensor[:, :2, 1:, :] - pts_tensor[:, :2, :-1, :]
        pts_tensor, motion_tensor = pts_tensor.to(self.device), motion_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model((pts_tensor, motion_tensor))   
            probs  = F.softmax(logits, dim=1)                  
            pred_i = int(torch.argmax(probs, dim=1).item())
            prob = float(probs[0, pred_i].item())
        return prob

