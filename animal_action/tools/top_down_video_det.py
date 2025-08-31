# -*- coding: utf-8 -*-
"""
Minimal, CPU-only, image-only pipeline:
  detector (YOLOX) -> pose (MMPose, optional draw) -> action (TSSTG)
Usage:
  python tools/top_down_video_det.py path/to/image.png [--no-skeleton]

Always overlays:
  - animal label + score (from YOLOX)
  - action label + score (from TSSTG)
Skeleton overlay is ON by default; pass --no-skeleton to hide.

Assumptions:
  - YOLOX model is COCO-trained (yolox-s by default).
  - ActionsEstLoader.py lives at repo root.
  - Pose config/ckpt are AP-10K HRNet-W32.
"""

import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from loguru import logger

# --- Make repo root importable so ActionsEstLoader.py at root can be found ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ActionsEstLoader import TSSTG  # noqa: E402
from mmpose.apis import inference_top_down_pose_model  # noqa: E402
from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.data.datasets import COCO_CLASSES, VOC_CLASSES  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import fuse_model, get_model_info, postprocess  # noqa: E402

# =========================
# Global constants (edit me)
# =========================

# YOLOX experiment/name and checkpoint (COCO, yolox-s by default)
MODEL_NAME = "yolox-s"
YOLOX_CKPT = "./target_detetion.pth"  # your detector checkpoint

# Pose (AP-10K HRNet-W32)
POSE_CONFIG = "./pose/hrnet_w32_ap10k_256_256.py"
POSE_CKPT = "./hrnet_w32_ap10k_256x256-18aac840_20211029.pth"

# Action model (TSSTG)
ACTION_CKPT = "ckpt/tsstg-model.pth"  # keep relative to project root

# Output directory (images will be saved here)
OUTPUT_DIR = "./YOLOX_outputs/vis_res"

# Inference/defaults
CONF_THRES = 0.5
NMS_THRES = 0.65
TEST_SIZE = 640


# =========================
# Helpers / pipeline blocks
# =========================

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def kpt2bbox(kpt, ex=20):
    return np.array(
        (
            kpt[:, 0].min() - ex,
            kpt[:, 1].min() - ex,
            kpt[:, 0].max() + ex,
            kpt[:, 1].max() + ex,
        )
    )


class Predictor:
    def __init__(self, model, exp, cls_names=VOC_CLASSES, fp16=False):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
            logger.info("Detector infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def to_pose_and_labels(self, output, img_info):
        """Return:
        - ret_bbox: list of dicts for MMPose [{bbox:[x1,y1,x2,y2,score], cls_id, cls_name, score}, ...]
        """
        ratio = img_info["ratio"]
        if output is None:
            return []

        output = output.cpu()
        bboxes = output[:, 0:4] / ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        ret_bbox = []
        np_bbox = bboxes.numpy().astype(int)
        np_score = scores.numpy()
        np_cls = cls.numpy().astype(int)

        for i in range(len(np_bbox)):
            score_i = float(np_score[i])
            cls_id_i = int(np_cls[i])
            cls_name_i = (
                self.cls_names[cls_id_i]
                if 0 <= cls_id_i < len(self.cls_names)
                else str(cls_id_i)
            )
            animal = {
                "bbox": np.concatenate((np_bbox[i], [score_i]), axis=0),
                "cls_id": cls_id_i,
                "cls_name": cls_name_i,
                "score": score_i,
            }
            ret_bbox.append(animal)
        return ret_bbox


def run_image(image_path: str, show_skeleton: bool):
    # ----- Build YOLOX (CPU) -----
    exp = get_exp(None, MODEL_NAME)
    exp.test_conf = CONF_THRES
    exp.nmsthre = NMS_THRES
    exp.test_size = (TEST_SIZE, TEST_SIZE)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    logger.info("loading detector checkpoint")
    ckpt = torch.load(YOLOX_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded detector checkpoint.")

    # Optional fuse (still CPU-safe)
    try:
        model = fuse_model(model)
    except Exception:
        pass

    predictor = Predictor(model, exp, VOC_CLASSES, fp16=False)

    # ----- Build pose (CPU) -----
    pose_model = init_pose_model(POSE_CONFIG, POSE_CKPT, device=torch.device("cpu"))
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the pose config.",
            DeprecationWarning,
        )
        dataset_info = None
    else:
        dataset_info = DatasetInfo(dataset_info)

    # ----- Build action model (CPU) -----
    action_pre = TSSTG(weight_file=ACTION_CKPT, device="cpu")

    # ----- Inference -----
    outputs, img_info = predictor.inference(image_path)
    if outputs is None or outputs[0] is None:
        raise RuntimeError("No detections found.")

    ret_bbox = predictor.to_pose_and_labels(outputs[0], img_info)
    if len(ret_bbox) == 0:
        raise RuntimeError("No detections above threshold.")

    frame = cv2.imread(image_path)

    # Pose (even if we don't draw skeletons, we still need kpts for the action label)
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        frame,
        ret_bbox,
        bbox_thr=0.3,
        format="xyxy",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None,
    )

    # Choose base image: with skeleton or not
    if show_skeleton:
        vis_img = vis_pose_result(
            pose_model,
            frame,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=0.2,
            radius=8,
            thickness=4,
            show=False,
        )
    else:
        vis_img = frame.copy()

    # Always draw YOLOX bounding boxes (even if skeleton is hidden)
    for rb in ret_bbox:
        # rb["bbox"] is [x1, y1, x2, y2, score]
        x1, y1, x2, y2, _ = rb["bbox"]
        # clamp to image bounds and int()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1] - 1, int(x2))
        y2 = min(frame.shape[0] - 1, int(y2))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw animal class name + score (always)
    for i, r in enumerate(pose_results):
        if "bbox" in r and r["bbox"] is not None:
            x1, y1, x2, y2 = list(map(int, r["bbox"][:4]))
            org_cls = (x1, max(0, y1 - 28))
            org_act = (x1, max(0, y1 - 8))
        else:
            # fallback to first keypoint
            kpt = r["keypoints"]
            org_cls = (int(kpt[0, 0]), max(0, int(kpt[0, 1]) - 28))
            org_act = (int(kpt[0, 0]), max(0, int(kpt[0, 1]) - 8))

        cls_name = ret_bbox[i].get("cls_name", "animal")
        cls_score = ret_bbox[i].get("score", 0.0)

        cv2.putText(
            vis_img,
            f"{cls_name} {cls_score:.2f}",
            org_cls,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # Action label (always). For single image, fake a short sequence.
        kpt = r["keypoints"]
        h, w = frame.shape[:2]
        seq = np.repeat(kpt[None, :, :], 30, axis=0)  # (T=30,V,3)
        probs = action_pre.predict(seq, (w, h))[0]
        label_idx = int(np.argmax(probs))
        label = action_pre.class_names[label_idx]
        score = float(probs[label_idx])

        cv2.putText(
            vis_img,
            f"{label} {score:.2f}",
            org_act,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, vis_img)
    logger.info(f"Saved: {out_path}")


# ===============
# CLI / entrypoint
# ===============


def make_parser():
    p = argparse.ArgumentParser("Animal action (image only, CPU)")
    p.add_argument("path", help="Path to a single image file")
    p.add_argument(
        "--no-skeleton",
        action="store_true",
        help="If set, do NOT draw skeletons (animal & action labels still drawn).",
    )
    return p


def main():
    args = make_parser().parse_args()

    # sanity checks
    ext = os.path.splitext(args.path)[1].lower()
    if ext not in IMAGE_EXT:
        raise ValueError(f"Unsupported image extension: {ext}")

    run_image(args.path, show_skeleton=not args.no_skeleton)


if __name__ == "__main__":
    main()
