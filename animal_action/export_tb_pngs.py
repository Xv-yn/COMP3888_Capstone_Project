import math
import os
import sys
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------- helpers ----------
def collect_scalars(logdir):
    """Return dict[tag] -> list of (step, wall_time, value) across all event files."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    out = defaultdict(list)
    for tag in sorted(ea.Tags().get("scalars", [])):
        for s in ea.Scalars(tag):
            out[tag].append((s.step, s.wall_time, s.value))
    return out


def dedupe_and_sort(points):
    """
    points: list[(step, wall_time, value)]
    - sort by (step, wall_time)
    - keep latest value per step (last wall_time)
    """
    points.sort(key=lambda x: (x[0], x[1]))
    last_by_step = OrderedDict()
    for step, wt, val in points:
        last_by_step[step] = (wt, val)  # overwrite -> keeps latest
    steps = list(last_by_step.keys())
    values = [last_by_step[s][1] for s in steps]
    return steps, values


def ema_smooth(values, alpha=0.2):
    if not values:
        return values
    sm = [values[0]]
    for v in values[1:]:
        sm.append(alpha * v + (1 - alpha) * sm[-1])
    return sm


def movavg(values, window=5):
    if window <= 1 or len(values) < 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


# ---------- main ----------
def export_one_run(logdir, smooth="ema", ema_alpha=0.2, ma_window=5):
    scalars = collect_scalars(logdir)
    outdir = os.path.join(logdir, "plots")
    os.makedirs(outdir, exist_ok=True)

    if not scalars:
        print(f"No scalar tags found in {logdir}")
        return
    print(f"Found {len(scalars)} tags:", sorted(scalars.keys()))

    for tag, pts in scalars.items():
        steps, values = dedupe_and_sort(pts)

        # optional smoothing (keeps raw too)
        if smooth == "ema":
            smoothed = ema_smooth(values, alpha=ema_alpha)
            sm_note = f"EMA α={ema_alpha}"
        elif smooth == "ma":
            smoothed = movavg(values, window=ma_window)
            sm_note = f"MA w={ma_window}"
        else:
            smoothed = None
            sm_note = None

        # make a nice, readable plot
        plt.figure(figsize=(7, 4.2))
        plt.plot(steps, values, marker="o", linewidth=1.5)
        if smoothed is not None and len(smoothed) > 2:
            plt.plot(steps, smoothed, linestyle="--", linewidth=2)

        # Labels/titles
        pretty = tag.replace("_", " ").replace("/", " / ")
        plt.title(pretty)
        plt.xlabel("step")
        plt.ylabel(pretty)

        # Extras for readability
        plt.grid(True, linewidth=0.6, alpha=0.6)
        # Show a small legend only if we plotted a smooth line
        if smoothed is not None and len(smoothed) > 2:
            plt.legend(["raw", sm_note], frameon=False)

        fname = f"{tag.replace('/','_')}.png"
        path = os.path.join(outdir, fname)
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        print("Saved", path)


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "YOLOX_outputs/yolox_s"
    # smooth can be "ema", "ma", or None
    export_one_run(run_dir, smooth="ema", ema_alpha=0.25)
