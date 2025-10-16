#!/usr/bin/env python3
"""
plots.py — Make training PNG charts from Ultralytics results.csv (fixed)

Usage
-----
# Point to a run directory (containing results.csv) or the CSV itself
python plots.py results/training/exp1
python plots.py results/training/exp1/results.csv

# With smoothing (rolling mean over 5 epochs)
python plots.py results/training/exp1 --smooth 5

Outputs
-------
<run_dir>/plots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_run_dir(path: Path) -> tuple[Path, Path]:
    p = path.resolve()
    if p.is_dir():
        csv_path = p / "results.csv"
        run_dir = p
    else:
        if p.name != "results.csv":
            sys.exit(f"[ERROR] Expected a run directory or results.csv, got: {p}")
        csv_path = p
        run_dir = p.parent
    if not csv_path.exists():
        sys.exit(f"[ERROR] results.csv not found at: {csv_path}")
    return run_dir, csv_path


def _savefig(out_dir: Path, name: str):
    out = out_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"[INFO] Wrote {out}")
    plt.close()


def _x(df: pd.DataFrame):
    return df["epoch"] if "epoch" in df.columns else range(len(df))


def _smooth(series: pd.Series, w: int | None):
    if w and w > 1:
        return series.rolling(window=w, min_periods=1).mean()
    return series


def _plot_masked_lines(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    out_dir: Path,
    name: str,
    smooth_win: int | None,
    mask_missing: bool = False,
):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    plt.figure(figsize=(8, 5))
    for c in cols:
        if mask_missing:
            mask = df[c].notna()
            if not mask.any():
                continue
            plt.plot(_x(df)[mask], _smooth(df.loc[mask, c], smooth_win), label=c)
        else:
            plt.plot(_x(df), _smooth(df[c], smooth_win), label=c)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _savefig(out_dir, name)


def _guess_loss_columns(df: pd.DataFrame, split: str) -> list[str]:
    candidates = [
        f"{split}/box_loss",
        f"{split}/cls_loss",
        f"{split}/dfl_loss",
        f"{split}/obj_loss",
        f"{split}/loss",
    ]
    bare = [c.replace(f"{split}/", "") for c in candidates]
    cols = [c for c in candidates if c in df.columns]
    cols += [c for c in df.columns if c in bare]
    seen, uniq = set(), []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _maybe_plot_losses(df: pd.DataFrame, out_dir: Path, smooth_win: int | None):
    train_cols = _guess_loss_columns(df, "train")
    val_cols = _guess_loss_columns(df, "val")

    # Combined losses: plot sums, but mask NaNs for val to avoid spikes
    has_train = any(c in df.columns for c in train_cols)
    has_val = any(c in df.columns for c in val_cols)

    if has_train or has_val:
        plt.figure(figsize=(8, 5))
        x_all = _x(df)

        if has_train:
            y_tr = df[train_cols].sum(axis=1, skipna=True)
            plt.plot(x_all, _smooth(y_tr, smooth_win), label="train(sum)")

        if has_val:
            y_val = df[val_cols].sum(axis=1, skipna=True)
            mask = y_val.notna()
            if mask.any():
                plt.plot(x_all[mask], _smooth(y_val[mask], smooth_win), label="val(sum)")

        plt.xlabel("epoch")
        plt.ylabel("sum of losses")
        plt.title("Loss (sum of available components)")
        plt.legend()
        _savefig(out_dir, "losses")

    # Detailed per-split curves (mask missing val rows)
    if train_cols:
        _plot_masked_lines(
            df,
            train_cols,
            "Train losses",
            "loss",
            out_dir,
            "train_losses",
            smooth_win,
            mask_missing=False,
        )
    if val_cols:
        _plot_masked_lines(
            df,
            val_cols,
            "Val losses",
            "loss",
            out_dir,
            "val_losses",
            smooth_win,
            mask_missing=True,
        )


def _maybe_plot_metrics(df: pd.DataFrame, out_dir: Path, smooth_win: int | None):
    metric_candidates = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP50",
        "metrics/mAP50-95",
        "val/precision",
        "val/recall",
        "val/mAP50",
        "val/mAP50-95",
    ]
    present = [c for c in metric_candidates if c in df.columns]
    if present:
        # Mask metrics like val (they often log sparsely)
        _plot_masked_lines(
            df,
            present,
            "Validation metrics",
            "metric",
            out_dir,
            "metrics",
            smooth_win,
            mask_missing=True,
        )


def _maybe_plot_lr(df: pd.DataFrame, out_dir: Path, smooth_win: int | None):
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    if lr_cols:
        _plot_masked_lines(
            df, lr_cols, "Learning rate(s)", "lr", out_dir, "lr", smooth_win, mask_missing=False
        )


def _maybe_plot_speed_and_size(df: pd.DataFrame, out_dir: Path, smooth_win: int | None):
    speed_cols = [c for c in df.columns if c.startswith("speed/")]
    if speed_cols:
        _plot_masked_lines(
            df, speed_cols, "Speed", "ms", out_dir, "speed", smooth_win, mask_missing=False
        )

    size_cols = [c for c in ("imgsz",) if c in df.columns]
    if size_cols:
        _plot_masked_lines(
            df, size_cols, "Image size", "value", out_dir, "size", smooth_win, mask_missing=False
        )


def _plot_misc_numeric(df: pd.DataFrame, out_dir: Path, smooth_win: int | None):
    # Skip columns already handled or not useful
    skip_prefixes = {"train/", "val/", "metrics/", "lr/", "speed/"}
    skip_exact = {
        "epoch",
        "imgsz",
        "time",
        "train/epoch",
        "memory",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "train/obj_loss",
        "train/loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/obj_loss",
        "val/loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP50",
        "metrics/mAP50-95",
    }

    for col in df.columns:
        if col in skip_exact or any(col.startswith(p) for p in skip_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Mask missing values to avoid fake lines
        mask = df[col].notna()
        if not mask.any():
            continue
        plt.figure(figsize=(7, 4.5))
        plt.plot(_x(df)[mask], _smooth(df.loc[mask, col], smooth_win), label=col)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title(col)
        plt.legend()
        safe = col.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")
        _savefig(out_dir, f"any_{safe}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot training curves from Ultralytics results.csv (NaN-aware)."
    )
    ap.add_argument("path", type=str, help="Run directory OR path to results.csv")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for PNGs (defaults to <run_dir>/plots)",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=None,
        help="Rolling average window (e.g., 5). Omit to disable.",
    )
    args = ap.parse_args()

    run_dir, csv_path = _ensure_run_dir(Path(args.path))
    out_dir = Path(args.out) if args.out else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read {csv_path}: {e}")
    if df.empty:
        sys.exit(f"[ERROR] {csv_path} is empty.")
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    _maybe_plot_losses(df, out_dir, args.smooth)
    _maybe_plot_metrics(df, out_dir, args.smooth)
    _maybe_plot_lr(df, out_dir, args.smooth)
    _maybe_plot_speed_and_size(df, out_dir, args.smooth)
    _plot_misc_numeric(df, out_dir, args.smooth)

    print(f"[INFO] Done. PNGs in: {out_dir}")


if __name__ == "__main__":
    main()
