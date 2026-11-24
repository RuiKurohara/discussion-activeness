# -*- coding: utf-8 -*-
"""
Compute the activeness models for all 50 discussion instances (Table 1).

Input
    data/pattern_dataset/P1/P1_01.txt ... P5/P5_10.txt

Output
    results/activity_log.txt       per-turn values of lambda1-lambda4 and the c(p), nu(p)
                                   of every proposition (used for Figure 4)
    results/summary_by_run.csv     final values of each instance (used for Figure 3)
    results/summary_by_pattern.csv mean and standard deviation over the ten instances of
                                   each pattern (Table 1). The standard deviation is the
                                   population standard deviation (ddof = 0).

Usage
    python evaluate_patterns.py [--device cuda|cpu] [--revision <HF model revision>]
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from activity_models import HFNLIBackend, NLIContentScorer, parse_pattern_file, run_stream

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pattern_dataset"
OUT_DIR = ROOT / "results"
PATTERNS = ["P1", "P2", "P3", "P4", "P5"]
KEYS = ["lambda1", "lambda2", "lambda3", "lambda4"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nli-model", default=HFNLIBackend.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    scorer = NLIContentScorer.zero_shot(model_name=args.nli_model,
                                        revision=args.revision, device=args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    finals = []  # (run name, {key: value})
    with (OUT_DIR / "activity_log.txt").open("w", encoding="utf-8") as log:
        for pattern in PATTERNS:
            for path in sorted((DATA_DIR / pattern).glob(f"{pattern}_*.txt")):
                for name, posts in parse_pattern_file(str(path)):
                    tracker = run_stream(name, posts, scorer=scorer,
                                         verbose=True, out=log)
                    run = path.stem
                    finals.append((run, {k: getattr(tracker, k) for k in KEYS}))
                    print(f"{run}: " + "  ".join(
                        f"{k}={getattr(tracker, k):.2f}" for k in KEYS))

    with (OUT_DIR / "summary_by_run.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"] + KEYS)
        for run, vals in finals:
            w.writerow([run] + [f"{vals[k]:.2f}" for k in KEYS])

    by_pattern = defaultdict(list)
    for run, vals in finals:
        by_pattern[run.split("_")[0]].append(vals)
    with (OUT_DIR / "summary_by_pattern.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "n"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "sd")])
        for pattern in PATTERNS:
            rows = by_pattern[pattern]
            cells = []
            for k in KEYS:
                xs = [r[k] for r in rows]
                cells += [f"{statistics.mean(xs):.2f}", f"{statistics.pstdev(xs):.2f}"]
            w.writerow([pattern, len(rows)] + cells)

    print(f"saved: {OUT_DIR / 'activity_log.txt'}")
    print(f"saved: {OUT_DIR / 'summary_by_run.csv'}")
    print(f"saved: {OUT_DIR / 'summary_by_pattern.csv'}")


if __name__ == "__main__":
    main()
