#!/usr/bin/env python3
"""
performance_test.py
-------------------
Runs modeling/predict.py on ALL images under a root folder (recursively),
records per-image latency, writes CSVs, prints summary, and generates plots.

Usage:
    python performance_test.py --images ./testing_set
Optional:
    --device cuda
    --option 1|2|3
    --no-skeleton
    --limit 20        # only test first N images
"""

import subprocess, json, csv, time, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- paths ---
HERE = Path(__file__).resolve().parent
PREDICT = (HERE.parent / "modeling" / "predict.py").resolve()
BENCH_DIR = HERE / "bench"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_recursively(root: Path):
    """Yield image paths recursively under root."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def run_predict(img_path: Path, device="cpu", option=3, show_skeleton=True):
    cmd = [
        sys.executable, str(PREDICT),
        "--option", str(option),
        "--image-path", str(img_path),
        "--device", device,
        "--show-skeleton" if show_skeleton else "--no-show-skeleton",
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    # fallback record
    perf = {"overall_ms": wall_ms, "meta": {"device": device, "option": option, "show_skeleton": show_skeleton}}

    if p.returncode == 0:
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PERF_JSON:")), None)
        if line:
            try:
                perf = json.loads(line[len("PERF_JSON:"):])
                perf.setdefault("overall_ms", wall_ms)
            except json.JSONDecodeError:
                pass
    return perf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Root folder containing images")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--option", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--no-skeleton", dest="skeleton", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N images")
    ap.set_defaults(skeleton=True)
    args = ap.parse_args()

    images_root = Path(args.images).resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images folder not found: {images_root}")
    if not PREDICT.exists():
        raise FileNotFoundError(f"predict.py not found: {PREDICT}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = BENCH_DIR / "perf_results.csv"

    imgs = list(list_images_recursively(images_root))
    if args.limit > 0:
        imgs = imgs[:args.limit]

    total = len(imgs)
    print(f"[INFO] Found {total} images to test under {images_root}")
    if total == 0:
        return

    rows = []
    for i, img in enumerate(imgs, 1):
        print(f"[{i}/{total}] {img.name}")
        perf = run_predict(img, args.device, args.option, args.skeleton)
        meta = perf.get("meta", {})
        rows.append({
            "image": str(img.relative_to(images_root)),
            "overall_ms": float(perf.get("overall_ms", 0.0)),
            "device": meta.get("device"),
            "option": meta.get("option"),
            "show_skeleton": meta.get("show_skeleton"),
            "num_cows": meta.get("num_cows"),
        })

    # Write per-image CSV
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Per-image CSV → {out_csv}")

    df = pd.DataFrame(rows)
    df["fps"] = 1000.0 / df["overall_ms"].clip(lower=1e-3)

    # --- Print summary statistics ---
    mean_ms = df["overall_ms"].mean()
    median_ms = df["overall_ms"].median()
    p95_ms = df["overall_ms"].quantile(0.95)
    min_ms = df["overall_ms"].min()
    max_ms = df["overall_ms"].max()
    fps_mean = df["fps"].mean()

    print("\n========= Performance Summary =========")
    print(f"Device:          {args.device}")
    print(f"Option:          {args.option}")
    print(f"Show skeleton:   {args.skeleton}")
    print("---------------------------------------")
    print(f"Mean latency:    {mean_ms:.2f} ms")
    print(f"Median latency:  {median_ms:.2f} ms")
    print(f"95th percentile: {p95_ms:.2f} ms")
    print(f"Min latency:     {min_ms:.2f} ms")
    print(f"Max latency:     {max_ms:.2f} ms")
    print(f"Average FPS:     {fps_mean:.2f}")
    print("=======================================\n")

    # Save summary to CSV
    summary_path = BENCH_DIR / "perf_summary.csv"
    df_summary = pd.DataFrame([{
        "device": args.device,
        "option": args.option,
        "show_skeleton": args.skeleton,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps_mean": fps_mean,
        "num_images": total
    }])
    df_summary.to_csv(summary_path, index=False)
    print(f"[OK] Summary CSV → {summary_path}")

    # --- Plots ---
    plt.figure()
    df["overall_ms"].plot(kind="hist", bins=30)
    plt.xlabel("Latency (ms)"); plt.title("Overall latency distribution")
    plt.tight_layout(); plt.savefig(BENCH_DIR / "latency_hist.png", dpi=170); plt.close()

    plt.figure()
    plt.plot(range(len(df)), df["overall_ms"], marker=".", linestyle="none")
    plt.ylabel("ms"); plt.xlabel("Image index")
    plt.title("Latency per image")
    plt.tight_layout(); plt.savefig(BENCH_DIR / "latency_per_image.png", dpi=170); plt.close()

    print(f"[OK] Plots → {BENCH_DIR}")

if __name__ == "__main__":
    main()
