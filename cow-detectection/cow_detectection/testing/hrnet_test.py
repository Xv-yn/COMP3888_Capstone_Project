#!/usr/bin/env python3
"""
hrnet_test.py (v4)
Runs HRNet inference from modeling/hrnet/hrnet_inference.py
on images in the testing folder (or any --source path).
Fixed to visible_ratio mode.
python hrnet_test.py --source ./yolo_results --thres 0.8
"""

import argparse, pickle, shutil, subprocess, sys
from pathlib import Path
import numpy as np

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def copy_images(src: Path, dst: Path):
    """Copy all images from testing folder into HRNet data/input folder."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images with {VALID_EXTS} under: {src}")
    for p in imgs:
        shutil.copy2(p, dst / p.name)
    return len(imgs)

def run_hrnet_inference(python_bin: str, script_path: Path):
    """Run hrnet_inference.py and stream logs live."""
    process = subprocess.Popen(
        [python_bin, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.stdout.close()
    ret = process.wait()
    if ret != 0:
        raise RuntimeError("hrnet_inference.py failed")

def visible_ratio_score(keypoints):
    """Compute ratio of visible joints (x,y != None)."""
    total = len(keypoints)
    if total == 0:
        return 0.0
    visible = sum(1 for kp in keypoints if kp.get("x") is not None and kp.get("y") is not None)
    return visible / total

def main():
    # --- Path setup ---
    testing_dir = Path(__file__).resolve().parent
    project_root = testing_dir.parent  # cow_detectection/
    hrnet_script = project_root / "modeling" / "hrnet" / "hrnet_inference.py"

    # HRNet uses its own relative paths (data/input/, results/)
    hrnet_root = hrnet_script.parent
    data_input = hrnet_root / "data" / "input"
    results_pkl = hrnet_root / "results" / "keypoints.pkl"

    ap = argparse.ArgumentParser(description="Run HRNet test on testing folder images.")
    ap.add_argument("--source", type=Path, default=testing_dir / "yolo_results",
                    help="Folder containing YOLO crop images (default: ./yolo_results)")
    ap.add_argument("--thres", type=float, default=0.60,
                    help="Skeleton passes if visible_ratio >= this threshold (default 0.8).")
    args = ap.parse_args()

    if not hrnet_script.exists():
        raise FileNotFoundError(f"Cannot find HRNet script at {hrnet_script}")

    # --- Step 1: prepare input ---
    n = copy_images(args.source, data_input)
    print(f"[prep] Copied {n} image(s) into {data_input}\n")

    # --- Step 2: run HRNet inference ---
    print(f"[run] Launching HRNet inference from: {hrnet_script}")
    run_hrnet_inference(sys.executable, hrnet_script)

    # --- Step 3: evaluate results ---
    if not results_pkl.exists():
        raise FileNotFoundError(f"Expected results at {results_pkl}, but not found.")

    with open(results_pkl, "rb") as f:
        results = pickle.load(f)

    accepted, total = 0, 0
    print("\n[eval] Evaluating skeleton quality...")
    for i, r in enumerate(results, 1):
        sk = visible_ratio_score(r.get("skeleton", []))
        passed = sk >= args.thres
        total += 1
        accepted += int(passed)
        status = "✓" if passed else "✗"
        print(f"  [{i:03d}] {r.get('image', '<unknown>'):25s}  visible_ratio={sk:5.2f}  {status}")

    rate = accepted / total if total else 0.0
    print("\n===== HRNet Test Summary =====")
    print(f"Images tested       : {total}")
    print(f"Accepted skeletons  : {accepted}")
    print(f"Acceptance rate     : {rate:.3f}")
    print(f"Mode                : visible_ratio (fixed)")
    print(f"Skeleton threshold  : {args.thres:.2f}")
    print(f"Results pickle      : {results_pkl}")

if __name__ == "__main__":
    main()
