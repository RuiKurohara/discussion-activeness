# -*- coding: utf-8 -*-
"""
Figure 3: box plots of the final activeness scores of the five patterns
(ten instances per pattern), grouped by activeness model.

Input : results/summary_by_run.csv   (written by evaluate_patterns.py)
Output: results/figures/pattern_comparison.png

Usage
    python plot_pattern_comparison.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "summary_by_run.csv"
OUT_PATH = ROOT / "results" / "figures" / "pattern_comparison.png"

BASE_FONT_SIZE = 16
matplotlib.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE + 2,
    "axes.labelsize": BASE_FONT_SIZE + 2,
    "xtick.labelsize": BASE_FONT_SIZE,
    "ytick.labelsize": BASE_FONT_SIZE,
    "legend.fontsize": BASE_FONT_SIZE,
    "legend.title_fontsize": BASE_FONT_SIZE + 1,
})

MODELS = [
    ("$\\lambda_{\\mathrm{baseline}}$", "lambda1"),
    ("$\\lambda_{\\mathrm{mutuality}}$", "lambda2"),
    ("$\\lambda_{\\mathrm{content}}$", "lambda3"),
    ("$\\lambda_{\\mathrm{mix}}$", "lambda4"),
]
MODEL_KEYS = [k for _, k in MODELS]

PATTERNS = ["P1", "P2", "P3", "P4", "P5"]
COLORS = {
    "P1": "#999999",
    "P2": "#1b9e77",
    "P3": "#e2725b",
    "P4": "#4a708b",
    "P5": "#7b2cbf",
}

# data[pattern][lambda_key] = [ten values]
data = defaultdict(lambda: defaultdict(list))
with CSV_PATH.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pattern = row["run"].split("_")[0]
        for key in MODEL_KEYS:
            data[pattern][key].append(float(row[key]))

for p in PATTERNS:
    for key in MODEL_KEYS:
        assert len(data[p][key]) == 10, (p, key, len(data[p][key]))

n_models = len(MODELS)
n_patterns = len(PATTERNS)
group_centers = np.arange(n_models)
box_width = 0.8 / n_patterns

fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

for i, pattern in enumerate(PATTERNS):
    offsets = group_centers - 0.4 + box_width * (i + 0.5)
    values = [data[pattern][key] for key in MODEL_KEYS]
    bp = ax.boxplot(values, positions=offsets, widths=box_width * 0.85,
                    patch_artist=True, showfliers=False, whis=(0, 100),
                    medianprops=dict(color="black", linewidth=1.5))
    for box in bp["boxes"]:
        box.set_facecolor(COLORS[pattern])
        box.set_edgecolor("white")
        box.set_alpha(0.9)
    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS[pattern])
    for cap in bp["caps"]:
        cap.set_color(COLORS[pattern])

ax.set_xticks(group_centers)
ax.set_xticklabels([label for label, _ in MODELS], fontsize=BASE_FONT_SIZE + 6)
ax.set_ylabel("Score")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

legend_handles = [mpatches.Patch(facecolor=COLORS[p], edgecolor="white", label=p)
                  for p in PATTERNS]
ax.legend(handles=legend_handles, title="Pattern", ncol=n_patterns,
          loc="upper left", frameon=True)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH)
print(f"saved: {OUT_PATH}")
