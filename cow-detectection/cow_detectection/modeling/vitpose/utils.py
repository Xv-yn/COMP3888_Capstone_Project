# utils.py
import json, math, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def gaussian_2d(sigma, size):
    x = np.arange(0, size, 1, float)
    y = x[:, None]
    x0 = y0 = size // 2
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def draw_gaussian(heatmap, center, sigma):
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape
    size = int(6 * sigma + 3)
    g = gaussian_2d(sigma, size)
    left, right = max(0, x - size//2), min(w, x + size//2 + 1)
    top, bottom = max(0, y - size//2), min(h, y + size//2 + 1)

    g_left = max(0, size//2 - x)
    g_top = max(0, size//2 - y)
    g_right = g_left + (right-left)
    g_bottom = g_top + (bottom-top)

    if left < right and top < bottom:
        heatmap[top:bottom, left:right] = np.maximum(
            heatmap[top:bottom, left:right],
            g[g_top:g_bottom, g_left:g_right]
        )
    return heatmap

def resize_with_aspect(img, target_h=256, target_w=192):
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

class CattleKeypointDataset(Dataset):
    """
    Expects a COCO-like annotation with:
    - images: [{id, file_name, ...}]
    - annotations: [{image_id, keypoints:[x,y,v]*K, num_keypoints:K, bbox:[x,y,w,h], ...}]
    """
    def __init__(self, images_dir, ann_file, num_keypoints=12,
                 input_hw=(256,192), heatmap_hw=(64,48), sigma=2.0, is_train=True):
        self.images_dir = Path(images_dir)
        self.num_keypoints = num_keypoints
        self.input_h, self.input_w = input_hw
        self.hm_h, self.hm_w = heatmap_hw
        self.sigma = sigma
        self.is_train = is_train

        data = json.loads(Path(ann_file).read_text())
        self.imgs = {img["id"]: img for img in data["images"]}
        self.anns = [ann for ann in data["annotations"] if ann.get("num_keypoints", 0) > 0]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_info = self.imgs[ann["image_id"]]
        img_path = self.images_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        assert img is not None, f"Missing image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simple center-crop around bbox then resize (you can replace with your YOLOX crop if available)
        if "bbox" in ann:
            x,y,w,h = ann["bbox"]
            x0,y0,x1,y1 = int(x),int(y),int(x+w),int(y+h)
            x0,y0 = max(0,x0), max(0,y0)
            x1,y1 = min(img.shape[1],x1), min(img.shape[0],y1)
            crop = img[y0:y1, x0:x1]
        else:
            crop = img

        crop = resize_with_aspect(crop, self.input_h, self.input_w)

        # Augment (very light)
        if self.is_train and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1, :])

        # Create target heatmaps
        heatmaps = np.zeros((self.num_keypoints, self.hm_h, self.hm_w), dtype=np.float32)

        kps = ann["keypoints"]  # [x,y,v]*K in original image space
        # We map keypoints into resized crop space. If you used bbox crop, re-map coordinates:
        # For simplicity, we assume bbox crop; adjust if not using.
        if "bbox" in ann:
            x,y,w,h = ann["bbox"]
            sx = self.input_w / max(w, 1e-6)
            sy = self.input_h / max(h, 1e-6)

        for k in range(self.num_keypoints):
            xk, yk, vk = kps[3*k:3*k+3]
            if vk < 1:  # not visible / missing
                continue
            if "bbox" in ann:
                xk = (xk - x) * sx
                yk = (yk - y) * sy
            else:
                # resized full image
                sx = self.input_w / img.shape[1]
                sy = self.input_h / img.shape[0]
                xk *= sx
                yk *= sy

            # Map to heatmap coords
            xhm = xk * (self.hm_w / self.input_w)
            yhm = yk * (self.hm_h / self.input_h)
            if 0 <= xhm < self.hm_w and 0 <= yhm < self.hm_h:
                heatmaps[k] = draw_gaussian(heatmaps[k], (xhm, yhm), self.sigma)

        # to tensors
        crop = torch.from_numpy(crop.transpose(2,0,1)).float() / 255.0
        heatmaps = torch.from_numpy(heatmaps)
        return crop, heatmaps
    
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

