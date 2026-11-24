# -*- coding: utf-8 -*-
"""
Figure 4: trajectories of the four activeness models over speaking turns
t = 1..6 (mean over ten instances; error bars = population standard deviation).

Input : results/activity_log.txt   (written by evaluate_patterns.py)
Output: results/figures/turn_progression.png

Usage
    python plot_turn_progression.py
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "results" / "activity_log.txt"
OUT_PATH = ROOT / "results" / "figures" / "turn_progression.png"

BASE_FONT_SIZE = 24
LEGEND_FONT_SIZE = 18
matplotlib.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE + 2,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE - 2,
    "ytick.labelsize": BASE_FONT_SIZE - 2,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.title_fontsize": LEGEND_FONT_SIZE + 1,
})

PATTERNS = ["P1", "P2", "P3", "P4", "P5"]
N_TURNS = 6
N_RUNS_EXPECTED = 10

STYLES = {
    "P1": dict(color="#999999", linestyle=":", marker="o"),
    "P2": dict(color="#1b9e77", linestyle="-.", marker="s"),
    "P3": dict(color="#e2725b", linestyle=":", marker="^"),
    "P4": dict(color="#4a708b", linestyle="-.", marker="D"),
    "P5": dict(color="#7b2cbf", linestyle="-", marker="*"),
}

MODELS = [
    ("lambda1", "$\\lambda_{\\mathrm{baseline}}$"),
    ("lambda2", "$\\lambda_{\\mathrm{mutuality}}$"),
    ("lambda3", "$\\lambda_{\\mathrm{content}}$"),
    ("lambda4", "$\\lambda_{\\mathrm{mix}}$"),
]
MODEL_KEYS = [k for k, _ in MODELS]

HEADER_RE = re.compile(r"^====\s*(P\d+)_\d+:")
ROW_RE = re.compile(r"^\s*\d+\*?\s*\|")


def parse_log(path: Path) -> dict:
    """data[pattern][turn] = [(l1, l2, l3, l4), ...] (one tuple per instance)."""
    data = defaultdict(lambda: defaultdict(list))
    current_pattern = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            hm = HEADER_RE.match(line)
            if hm:
                current_pattern = hm.group(1)
                continue
            if current_pattern and ROW_RE.match(line):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 9:
                    continue
                turn = int(re.sub(r"\D", "", parts[0]))
                try:
                    l1, l2, l3, l4 = (float(parts[5]), float(parts[6]),
                                      float(parts[7]), float(parts[8]))
                except ValueError:
                    continue
                data[current_pattern][turn].append((l1, l2, l3, l4))
    return data


def main():
    raw = parse_log(LOG_PATH)

    stats = {}
    for pattern in PATTERNS:
        per_model = {k: {"mean": [], "std": []} for k in MODEL_KEYS}
        for t in range(1, N_TURNS + 1):
            runs = raw[pattern].get(t, [])
            if len(runs) != N_RUNS_EXPECTED:
                print(f"warning: {pattern} t={t} has {len(runs)} runs "
                      f"(expected {N_RUNS_EXPECTED})")
            arr = np.array(runs)  # shape (n_runs, 4)
            for i, key in enumerate(MODEL_KEYS):
                col = arr[:, i]
                per_model[key]["mean"].append(col.mean())
                per_model[key]["std"].append(col.std())  # population SD (ddof=0)
        stats[pattern] = per_model

    turns = np.arange(1, N_TURNS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)

    for ax, (key, title) in zip(axes.flat, MODELS):
        for pattern in PATTERNS:
            s = stats[pattern][key]
            ax.errorbar(turns, np.array(s["mean"]), yerr=np.array(s["std"]),
                        label=pattern, markersize=11, linewidth=2.5, capsize=5,
                        elinewidth=1.5, **STYLES[pattern])
        ax.set_title(title)
        ax.set_xlabel("Discussion turn $t$")
        ax.set_ylabel("Score")
        ax.set_xticks(turns)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0, 0].text(
        0.05, 0.75,
        "Identical across all patterns\n(since $\\lambda_{\\mathrm{baseline}}(t)=t$)",
        transform=axes[0, 0].transAxes, fontsize=BASE_FONT_SIZE - 4,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pattern (solid line = mean, bars = SD)",
               ncol=len(PATTERNS), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
