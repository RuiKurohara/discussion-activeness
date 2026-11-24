# -*- coding: utf-8 -*-
"""
Checks the worked examples of Sections 3.2 and 3.4 of the paper. The content
scores q_m are given explicitly, so no NLI model is needed.

Usage
    python tests/test_worked_examples.py      (or: python -m pytest tests)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity_models import ActivityTracker, gini  # noqa: E402


def run(order, qs=None):
    tr = ActivityTracker()
    for i, speaker in enumerate(order):
        tr.add_post(speaker, "text", q=1.0 if qs is None else qs[i])
    return tr


def test_gini():
    assert gini({"A": 2, "B": 2, "C": 2}) == 0.0
    assert gini({"A": 6}) == 0.0
    assert abs(gini({"A": 4, "B": 1, "C": 1}) - 1 / 3) < 1e-12


def test_mutuality_abcabc():
    # Section 3.2: A,B,C,A,B,C -> sum I_m = 1+1+1+2+2+2 = 9, G = 0
    tr = run("ABCABC")
    assert tr.G == 0.0
    assert tr.lambda2 == 9.0


def test_mutuality_aabbcc():
    # Section 3.2: A,A,B,B,C,C is compressed to A,B,C -> sum I_m = 3, G = 0
    tr = run("AABBCC")
    assert tr.lambda2 == 3.0


def test_mix_agreement_only():
    # Section 3.4: A*,B,C,A*,B,C with q_m = 1 for A and 0 for B and C
    tr = run("ABCABC", qs=[1, 0, 0, 1, 0, 0])
    assert tr.lambda4 == 3.0
    assert tr.lambda2 == 9.0


def test_monologue_mix_equals_content():
    # A monologue is compressed to one post with I_m = 1 and G = 0,
    # so lambda_mix equals lambda_content.
    qs = [0.5, 1.0, 0.25, 0.75, 1.0, 0.5]
    tr = run("AAAAAA", qs=qs)
    assert tr.lambda2 == 1.0
    assert abs(tr.lambda4 - tr.lambda3) < 1e-12
    assert abs(tr.lambda3 - sum(qs)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
