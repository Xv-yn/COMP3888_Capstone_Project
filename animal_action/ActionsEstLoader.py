import numpy as np
import torch

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose.pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) 'cpu' or 'cuda' / 'cuda:0'
    """

    def __init__(self, weight_file="ckpt/tsstg-model.pth", device="cpu"):
        self.graph_args = {"strategy": "spatial"}
        self.class_names = ["Stand", "Walk", "Run", "Lay", "Eat"]
        self.num_class = len(self.class_names)

        # pick device safely
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # build model
        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(
            self.device
        )

        # load checkpoint safely to CPU (works even on GPU machines)
        ckpt = torch.load(weight_file, map_location=torch.device("cpu"))

        # accept either raw state_dict or wrapped dicts
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt  # already a state_dict-like mapping
        else:
            state_dict = ckpt

        # load with strict=False to be resilient to minor key diffs
        self.model.load_state_dict(state_dict, strict=False)

        # move to target device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pts, image_size):
        """Predict action probs from single-person skeleton over time.
        pts: (t, v, c) with c=(x,y,score); image_size=(W,H)
        returns: numpy array of class probabilities
        """

        pts = np.repeat(pts, 60, axis=0)

        pts[:, :, :2] = normalize_points_with_size(
            pts[:, :, :2], image_size[0], image_size[1]
        )
        pts[:, :, :2] = scale_pose(pts[:, :, :2])

        # add midpoint joint between 3 and 4 as extra node
        pts = np.concatenate(
            (pts, np.expand_dims((pts[:, 3, :] + pts[:, 4, :]) / 2, 1)), axis=1
        )

        pts = torch.tensor(pts, dtype=torch.float32)  # (T, V, C)
        pts = pts.permute(2, 0, 1)[None, :]  # (N=1, C, T, V)

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]  # motion stream (x,y)

        pts = pts.to(self.device)
        mot = mot.to(self.device)

        with torch.no_grad():
            out = self.model((pts, mot))  # logits or probs depending on model

        return out.detach().cpu().numpy()
