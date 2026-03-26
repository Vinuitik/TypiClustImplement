"""
Generate 4 AL benchmark plots with mean ± std bands accumulated across runs.

  Plot 1 — eval,  budgets [20, 30, 40, 50, 60]         (small range)
  Plot 2 — mlp,   budgets [20, 30, 40, 50, 60]         (small range)
  Plot 3 — eval,  budgets [50, 100, 150, 200, 250, 300] (medium range)
  Plot 4 — mlp,   budgets [50, 100, 150, 200, 250, 300] (medium range)

Eval files  : datasets/al_eval_results*.json  (excluding *resnet_mlp*)
MLP files   : datasets/al_eval_results_resnet_mlp*.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

PLOTS = [
    dict(tag="eval", budgets=[20, 30, 40, 50, 60],              title="ResNet eval  —  small budgets (20–60)",    out="plot1_eval_small.png"),
    dict(tag="mlp",  budgets=[20, 30, 40, 50, 60],              title="MLP head     —  small budgets (20–60)",    out="plot2_mlp_small.png"),
    dict(tag="eval", budgets=[50, 100, 150, 200, 250, 300],      title="ResNet eval  —  medium budgets (50–300)",  out="plot3_eval_medium.png"),
    dict(tag="mlp",  budgets=[50, 100, 150, 200, 250, 300],      title="MLP head     —  medium budgets (50–300)",  out="plot4_mlp_medium.png"),
]

# 13 visually distinct colors for the methods
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#808000",
]

# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
def load_files(paths: list[Path]) -> dict[str, dict[int, list[float]]]:
    """Returns {method: {budget: [acc, acc, ...]}}"""
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for entry in json.loads(p.read_text(encoding="utf-8")):
            data[entry["method"]][int(entry["budget"])].append(float(entry["test_acc"]))
    return data


def collect(tag: str) -> dict[str, dict[int, list[float]]]:
    if tag == "eval":
        paths = [p for p in DATASETS.glob("al_eval_results*.json") if "resnet_mlp" not in p.name]
    else:
        paths = list(DATASETS.glob("al_eval_results_resnet_mlp*.json"))
    if not paths:
        raise FileNotFoundError(f"No files found for tag '{tag}' in {DATASETS}")
    print(f"[{tag}] loading {[p.name for p in paths]}")
    return load_files(paths)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(cfg: dict, data: dict[str, dict[int, list[float]]]) -> None:
    budgets = cfg["budgets"]
    methods = sorted(data.keys())

    # Extra top margin so the above-axes legend fits
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.78)

    plotted_handles = []
    plotted_labels  = []

    for i, method in enumerate(methods):
        color = COLORS[i % len(COLORS)]
        xs, means, stds = [], [], []

        for b in budgets:
            values = data[method].get(b)
            if not values:
                continue
            xs.append(b)
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=0))

        if not xs:
            continue

        xs    = np.array(xs)
        means = np.array(means)
        stds  = np.array(stds)

        (line,) = ax.plot(
            xs, means,
            marker="o", color=color, linewidth=1.8, markersize=4,
        )
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color)

        plotted_handles.append(line)
        plotted_labels.append(method.replace("_", " "))

    ax.set_xticks(budgets)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Labeled budget", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legend above the plot area, two rows, labels next to their color patches
    ncols = min(7, len(plotted_labels))
    ax.legend(
        plotted_handles, plotted_labels,
        fontsize=7.5,
        ncol=ncols,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0,
        frameon=True,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    out_path = OUT_DIR / cfg["out"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cache: dict[str, dict] = {}
    for cfg in PLOTS:
        tag = cfg["tag"]
        if tag not in cache:
            cache[tag] = collect(tag)
        print(f"Generating: {cfg['out']}")
        make_plot(cfg, cache[tag])
    print("Done.")
