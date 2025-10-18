# dataset_cattle.py
from pathlib import Path
import json, cv2, numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms

class CattleKeypointDataset(Dataset):
    def __init__(self, images_root, ann_file,
                 input_hw=(256,192), heatmap_hw=(64,48),
                 sigma=2.0, is_train=True):
        self.images_root = Path(images_root)
        self.input_h, self.input_w = input_hw
        self.hm_h, self.hm_w = heatmap_hw
        self.sigma = sigma
        self.is_train = is_train

        d = json.loads(Path(ann_file).read_text())
        cat = next((c for c in d["categories"] if "keypoints" in c and c["keypoints"]), None)
        if cat is None:
            raise ValueError("No 'categories[].keypoints' found in annotation.")
        self.keypoint_names = list(cat["keypoints"])
        self.skeleton_pairs_1based = [tuple(pair) for pair in cat.get("skeleton", [])]
        self.num_keypoints = len(self.keypoint_names)
        self.imgs = {img["id"]: img for img in d["images"]}

        # accept samples that have a full keypoint list even if 'num_keypoints' is missing
        K = self.num_keypoints
        self.anns = [a for a in d["annotations"] if len(a.get("keypoints", [])) >= 3 * K]
        if len(self.anns) == 0:
            raise ValueError("No annotations with keypoints found.")

        # build filename -> path map (recursive), case-insensitive
        all_imgs = [p for p in self.images_root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        self._filename_to_path = {p.name: p for p in all_imgs}
        self._filename_to_path.update({p.name.lower(): p for p in all_imgs})

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_h, self.input_w)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        """
        Robust __getitem__:
        - Resolve path by name / relative path
        - Skip missing/unreadable images or malformed keypoints
        - Safe bbox → crop → resize → heatmaps
        """
        num_samples = len(self.anns)
        assert num_samples > 0, "Dataset has no annotations."

        while True:
            ann = self.anns[idx % num_samples]
            img_info = self.imgs[ann["image_id"]]

            # ---- resolve image path ----
            fname = str(img_info.get("file_name", "")).replace("\\", "/")
            base  = Path(fname).name
            path = (self._filename_to_path.get(base)
                    or self._filename_to_path.get(base.lower())
                    or (self.images_root / fname))
            if not (path and Path(path).exists()):
                print(f"[WARN] Skipping missing image: '{fname}' under '{self.images_root}'")
                idx += 1
                continue

            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                print(f"[WARN] Cannot read image: {path}, skipping")
                idx += 1
                continue

            print(f"[INFO] Loaded image: {path.name}  shape={img_bgr.shape}")
            
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ---- keypoints sanity ----
            kps = ann.get("keypoints", [])
            if len(kps) < 3 * self.num_keypoints:
                print(f"[WARN] Bad keypoints for image_id={ann.get('image_id')}, skipping")
                idx += 1
                continue

            # ---- bbox → crop (guard) ----
            x, y, w, h = ann.get("bbox", [0, 0, img.shape[1], img.shape[0]])
            if w < 2 or h < 2:
                x, y, w, h = 0, 0, img.shape[1], img.shape[0]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                print(f"[WARN] Empty crop for {path}, using full image")
                crop = img

            x_t = self.tf(crop)  # [3,256,192]

            # ---- targets: K heatmaps ----
            heatmaps = np.zeros((self.num_keypoints, self.hm_h, self.hm_w), dtype=np.float32)

            sx = self.input_w / max(float(x1 - x0), 1e-6)
            sy = self.input_h / max(float(y1 - y0), 1e-6)
            for k in range(self.num_keypoints):
                xk, yk, vk = kps[3*k:3*k+3]
                if vk < 1:
                    continue
                xk_in = (xk - x0) * sx
                yk_in = (yk - y0) * sy
                xhm = xk_in * (self.hm_w / self.input_w)
                yhm = yk_in * (self.hm_h / self.input_h)
                if 0 <= xhm < self.hm_w and 0 <= yhm < self.hm_h:
                    heatmaps[k] = self._draw_gaussian(heatmaps[k], (xhm, yhm), sigma=2.0)

            return x_t, torch.from_numpy(heatmaps)

    def get_schema(self):
        """
        Returns a consistent schema dict for training/inference & drawing.
        """
        return {
            "num_keypoints": self.num_keypoints,                     # K
            "keypoint_names": list(self.keypoint_names),             # ['1','2',...]
            "skeleton_0based": [(a-1, b-1) for (a, b) in self.skeleton_pairs_1based],  # edges
        }


    # ---- heatmap helpers ----
    @staticmethod
    def _gaussian_2d(sigma, size):
        x = np.arange(0, size, 1, float); y = x[:, None]
        x0 = y0 = size // 2
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    @staticmethod
    def _draw_gaussian(heatmap, center, sigma):
        x, y = int(center[0]), int(center[1])
        h, w = heatmap.shape
        size = int(6 * sigma + 3)
        g = CattleKeypointDataset._gaussian_2d(sigma, size)
        left, right = max(0, x - size//2), min(w, x + size//2 + 1)
        top, bottom = max(0, y - size//2), min(h, y + size//2 + 1)
        g_left = max(0, size//2 - x); g_top = max(0, size//2 - y)
        g_right = g_left + (right-left); g_bottom = g_top + (bottom-top)
        if left < right and top < bottom:
            heatmap[top:bottom, left:right] = np.maximum(
                heatmap[top:bottom, left:right],
                g[g_top:g_bottom, g_left:g_right]
            )
        return heatmap
