# -*- coding: utf-8 -*-
"""
Streaming implementation of the discussion-activeness models.

This module implements the four activeness models compared in the paper
"Modeling Discussion Activeness in Online Forums Based on Post Mutuality and
Content Quality" (ICAART). The names used in the code and in the paper are:

    code       paper                    definition
    lambda1    lambda_baseline (Eq. 8)  |M(t)|                               (post count)
    lambda2    lambda_mutuality (Eq. 3) (1 - G(t)) * sum_{b_m in M'(t)} I_m
    lambda3    lambda_content (Eq. 6)   sum_{x_m in M(t)} q_m
    lambda4    lambda_mix (Eq. 7)       (1 - G(t)) * sum_{b_m in M'(t)} I_m q_m

Notation
    M(t)  : posts observed up to time t (uncompressed)
    M'(t) : compressed sequence in which consecutive posts by the same speaker
            are merged into a single post
    I_m   : mutuality score on M'(t) (Eq. 1)
            I_m = idx(b_m) - idx(prev(b_m)) - 1, and I_m = 1 for a first contribution
    q_m   : content score (Eq. 4), q_m = sum_{p in P_m} c(p) * nu(p)
            c(p)  : specificity of proposition p, in [0, 1]
            nu(p) : novelty (Eq. 5), 1 - max_{p' in H_{m-1}} P(Rephrase | p, p'),
                    and nu(p) = 1 if H_{m-1} is empty
    G(t)  : Gini coefficient of the uncompressed post counts per speaker (Eq. 2)
            G = sum_a sum_b |n_a - n_b| / (2 |U| sum_a n_a)

Posts are assumed to arrive one at a time as (speaker, text) pairs. Every call
to ActivityTracker.add_post() updates all four models incrementally.

Content scorer used in the paper
    c(p) and P(Rephrase | p, p') are computed by zero-shot natural language
    inference (NLI) with the multilingual model mDeBERTa-v3-XNLI
    (Laurer et al., 2024), without additional training:
      * c(p): the entailment probabilities of two hypotheses ("concrete
        proposal/claim" vs. "only agreement/acknowledgment") are normalized
        against each other (ZeroShotSpecificity).
      * P(Rephrase | p, p') is approximated by the product of the
        bidirectional entailment probabilities P(p => p') * P(p' => p)
        (NLIRephrase).
    Each post is split into sentences, and each sentence is treated as one
    proposition (sentence_segmenter).

Usage
    python activity_models.py                          # built-in toy example, q_m given explicitly
    python activity_models.py data/pattern_dataset/P3/P3_01.txt -v
                                                       # score one discussion with the NLI scorer
To reproduce all results reported in the paper, use evaluate_patterns.py.

Requirements for the NLI scorer: torch, transformers, sentencepiece.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Gini coefficient (Eq. 2), computed from the uncompressed post counts
# ---------------------------------------------------------------------------

def gini(counts: dict[str, int]) -> float:
    """Return the Gini coefficient G for {speaker: number of posts}.

    G = sum_a sum_b |n_a - n_b| / (2 |U| sum_a n_a).
    Returns 0 when there are no speakers or no posts.
    """
    values = list(counts.values())
    n_users = len(values)
    total = sum(values)
    if n_users == 0 or total == 0:
        return 0.0
    diff_sum = sum(abs(a - b) for a in values for b in values)
    return diff_sum / (2 * n_users * total)


# ---------------------------------------------------------------------------
# Interface of the content scorer that computes q_m
# ---------------------------------------------------------------------------

class SpecificityScorer(Protocol):
    """Computes the content score q_m of a newly arrived post.

    score(text, history_texts) -> q_m
      text          : text of the new post
      history_texts : texts of all earlier posts (chronological, uncompressed)
    """

    def score(self, text: str, history_texts: list[str]) -> float: ...


# ---------------------------------------------------------------------------
# Proposition segmentation P_m (post -> list of propositions)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")


def sentence_segmenter(text: str) -> list[str]:
    """Split a post into sentences; each sentence is treated as one proposition."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# NLI backend (HuggingFace transformers; imported lazily)
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class HFNLIBackend:
    """Zero-shot NLI model: (premise, hypothesis) pairs -> P(entailment)."""

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 revision: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 32,
                 max_length: int = 256) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision)
        self.model.to(self.device).eval()
        # Index of the "entailment" label, taken from id2label
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self._ent_idx = next(i for i, l in id2label.items() if "entail" in l)

    def entail_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                enc = self.tokenizer(
                    [a for a, _ in batch], [b for _, b in batch],
                    truncation=True, max_length=self.max_length,
                    padding=True, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
                out.extend(probs[:, self._ent_idx].tolist())
        return out


# ---------------------------------------------------------------------------
# c(p): specificity of a proposition (continuous value in [0, 1])
# ---------------------------------------------------------------------------

class ZeroShotSpecificity:
    """Zero-shot NLI specificity c(p).

    The proposition p is used as the premise. The entailment probabilities of
    one hypothesis per class are normalized against each other:
        c(p) = P(p => HYP_SPECIFIC) / (P(p => HYP_SPECIFIC) + P(p => HYP_GENERIC)).
    """

    HYP_SPECIFIC = "This makes a concrete proposal, claim, or new piece of information."
    HYP_GENERIC = "This is just agreement or acknowledgment, with no new information."

    def __init__(self, nli: HFNLIBackend,
                 hyp_specific: str = HYP_SPECIFIC,
                 hyp_generic: str = HYP_GENERIC) -> None:
        self.nli = nli
        self.hyp_specific = hyp_specific
        self.hyp_generic = hyp_generic

    def scores(self, props: Sequence[str]) -> list[float]:
        if not props:
            return []
        pairs = [(p, self.hyp_specific) for p in props] + \
                [(p, self.hyp_generic) for p in props]
        ent = self.nli.entail_probs(pairs)
        n = len(props)
        out = []
        for i in range(n):
            e_spec, e_gen = ent[i], ent[n + i]
            denom = e_spec + e_gen
            out.append(e_spec / denom if denom > 0 else 0.5)
        return out


# ---------------------------------------------------------------------------
# P(Rephrase | p, p') and novelty nu(p)
# ---------------------------------------------------------------------------

class NLIRephrase:
    """Rephrasing probability approximated by bidirectional entailment.

        P(Rephrase | p, p') ~= P(p => p') * P(p' => p)

    A full paraphrase entails in both directions (value close to 1); a
    specialization or elaboration entails in one direction only; unrelated or
    new content entails in neither direction (value close to 0).
    """

    def __init__(self, nli: HFNLIBackend) -> None:
        self.nli = nli

    def probs(self, p: str, history: Sequence[str]) -> list[float]:
        if not history:
            return []
        pairs = [(p, h) for h in history] + [(h, p) for h in history]
        ent = self.nli.entail_probs(pairs)
        n = len(history)
        return [ent[i] * ent[n + i] for i in range(n)]


@dataclass
class PropositionScore:
    """Per-proposition breakdown, for inspection."""
    proposition: str
    c: float
    nu: float
    contribution: float  # c * nu


class NLIContentScorer:
    """Content scorer q_m = sum_p c(p) nu(p) used in the paper (zero-shot NLI)."""

    def __init__(self, specificity: ZeroShotSpecificity, rephrase: NLIRephrase,
                 segmenter=sentence_segmenter) -> None:
        self.specificity = specificity
        self.rephrase = rephrase
        self.segmenter = segmenter
        self._history_props: list[str] = []   # H: all earlier propositions
        self.last_detail: list[PropositionScore] = []

    @classmethod
    def zero_shot(cls, model_name: str = HFNLIBackend.DEFAULT_MODEL,
                  revision: Optional[str] = None,
                  device: Optional[str] = None,
                  batch_size: int = 32) -> "NLIContentScorer":
        nli = HFNLIBackend(model_name, revision=revision, device=device,
                           batch_size=batch_size)
        return cls(specificity=ZeroShotSpecificity(nli), rephrase=NLIRephrase(nli))

    def score(self, text: str, history_texts: list[str]) -> float:
        """Return q_m for one post.

        Novelty is computed against the internal proposition history H rather
        than against history_texts, because nu(p) is defined per proposition.
        """
        props = self.segmenter(text)
        if not props:
            self.last_detail = []
            return 0.0

        c_scores = self.specificity.scores(props)
        H = self._history_props          # H_{m-1}: propositions before this post

        q = 0.0
        detail: list[PropositionScore] = []
        for p, c in zip(props, c_scores):
            nu = 1.0 - max(self.rephrase.probs(p, H)) if H else 1.0
            contrib = c * nu
            q += contrib
            detail.append(PropositionScore(p, c, nu, contrib))

        # P_m is added to H after the post has been scored.
        self._history_props.extend(props)
        self.last_detail = detail
        return q

    def reset(self) -> None:
        """Clear the proposition history. Call this before each new discussion."""
        self._history_props.clear()
        self.last_detail = []


# ---------------------------------------------------------------------------
# Activeness tracker (incremental update)
# ---------------------------------------------------------------------------

@dataclass
class CompressedMessage:
    """One element b_m of the compressed sequence M'(t)."""
    idx: int          # position in M'(t), starting at 1
    speaker: str
    I: int            # mutuality score I_m (fixed when b_m is created)
    q: float = 0.0    # sum of q_m of the original posts merged into b_m


@dataclass
class Snapshot:
    """Values of all models after add_post()."""
    t: int                    # number of posts processed so far, |M(t)|
    speaker: str
    text: str
    I_m: int                  # I of the new compressed post (0 if merged into the previous one)
    q_m: float
    merged: bool              # True if merged into the previous post by consecutive-post compression
    G: float
    lambda1: float            # lambda_baseline
    lambda2: float            # lambda_mutuality
    lambda3: float            # lambda_content
    lambda4: float            # lambda_mix


class ActivityTracker:
    """Receives posts one at a time and updates lambda1 to lambda4 incrementally.

    If q is passed to add_post(), it is used as q_m; otherwise q_m is computed
    by the scorer. A scorer is needed only when q is omitted.
    """

    def __init__(self, scorer: Optional[SpecificityScorer] = None) -> None:
        self.scorer = scorer
        # Uncompressed sequence M(t)
        self._texts: list[str] = []
        self._counts: dict[str, int] = {}      # posts per speaker (for G, uncompressed)
        # Compressed sequence M'(t)
        self._compressed: list[CompressedMessage] = []
        self._last_idx_of: dict[str, int] = {}  # latest idx of each speaker in M'(t)
        # Running sums
        self._sum_I: float = 0.0    # sum I_m       (lambda2)
        self._sum_q: float = 0.0    # sum q_m, uncompressed (lambda3)
        self._sum_Iq: float = 0.0   # sum I_m q_m, compressed (lambda4)

    @property
    def n_posts(self) -> int:
        return len(self._texts)

    @property
    def G(self) -> float:
        return gini(self._counts)

    @property
    def lambda1(self) -> float:
        return float(self.n_posts)

    @property
    def lambda2(self) -> float:
        return (1.0 - self.G) * self._sum_I

    @property
    def lambda3(self) -> float:
        return self._sum_q

    @property
    def lambda4(self) -> float:
        return (1.0 - self.G) * self._sum_Iq

    def add_post(self, speaker: str, text: str, q: Optional[float] = None) -> Snapshot:
        """Add one post and return the updated values of all models."""
        if q is not None:
            q_m = float(q)
        elif self.scorer is not None:
            q_m = float(self.scorer.score(text, self._texts))  # history = earlier posts
        else:
            raise ValueError("Pass q explicitly or construct the tracker with a scorer.")

        # Uncompressed sequence M(t)
        self._texts.append(text)
        self._counts[speaker] = self._counts.get(speaker, 0) + 1
        self._sum_q += q_m                      # lambda3: sum over all uncompressed posts

        # Consecutive-post compression and mutuality score on M'(t)
        if self._compressed and self._compressed[-1].speaker == speaker:
            # Consecutive post: merged into the previous compressed post.
            # I is unchanged; only q is added (q of b_m = sum of the merged posts).
            merged = True
            cm = self._compressed[-1]
            cm.q += q_m
            self._sum_Iq += cm.I * q_m
            I_m = 0                             # for display: no new mutuality
        else:
            merged = False
            idx = len(self._compressed) + 1
            if speaker in self._last_idx_of:
                I_m = idx - self._last_idx_of[speaker] - 1
            else:
                I_m = 1                         # first contribution
            cm = CompressedMessage(idx=idx, speaker=speaker, I=I_m, q=q_m)
            self._compressed.append(cm)
            self._last_idx_of[speaker] = idx
            self._sum_I += I_m
            self._sum_Iq += I_m * q_m

        return Snapshot(
            t=self.n_posts, speaker=speaker, text=text,
            I_m=I_m, q_m=q_m, merged=merged, G=self.G,
            lambda1=self.lambda1, lambda2=self.lambda2,
            lambda3=self.lambda3, lambda4=self.lambda4,
        )


# ---------------------------------------------------------------------------
# Reading discussion files ("[m=1] Alice: ..." format)
# ---------------------------------------------------------------------------

_POST_RE = re.compile(r"^\[m=(\d+)\]\s*([^:]+):\s*(.*)$")
_HEADER_RE = re.compile(r"^====\s*(.+?)\s*====$")


def parse_pattern_file(path: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return [(discussion name, [(speaker, text), ...]), ...] for a discussion file."""
    patterns: list[tuple[str, list[tuple[str, str]]]] = []
    name, posts = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            h = _HEADER_RE.match(line)
            if h:
                if name is not None:
                    patterns.append((name, posts))
                name, posts = h.group(1), []
                continue
            m = _POST_RE.match(line)
            if m and name is not None:
                posts.append((m.group(2).strip(), m.group(3).strip()))
    if name is not None:
        patterns.append((name, posts))
    return patterns


# ---------------------------------------------------------------------------
# Running one discussion
# ---------------------------------------------------------------------------

def run_stream(name: str, posts: Iterable[tuple[str, str]],
               scorer: Optional[SpecificityScorer] = None,
               qs: Optional[Sequence[float]] = None,
               verbose: bool = False, out=sys.stdout) -> ActivityTracker:
    """Feed one discussion post by post and print the values at every turn.

    The scorer's proposition history is reset first, so that propositions of
    an earlier discussion do not lower nu(p) in this one.
    """
    if scorer is not None and hasattr(scorer, "reset"):
        scorer.reset()
    tracker = ActivityTracker(scorer=scorer)
    print(f"\n==== {name} ====", file=out)
    print(f"{'t':>2} | {'speaker':<8} | {'I_m':>3} | {'q_m':>5} | "
          f"{'G':>5} | {'λ1':>4} | {'λ2':>6} | {'λ3':>6} | {'λ4':>6}", file=out)
    print("-" * 66, file=out)
    for i, (speaker, text) in enumerate(posts):
        s = tracker.add_post(speaker, text, q=None if qs is None else qs[i])
        mark = "*" if s.merged else " "
        print(f"{s.t:>2}{mark}| {s.speaker:<8} | {s.I_m:>3} | {s.q_m:>5.2f} | "
              f"{s.G:>5.3f} | {s.lambda1:>4.0f} | {s.lambda2:>6.2f} | "
              f"{s.lambda3:>6.2f} | {s.lambda4:>6.2f}", file=out)
        if verbose and qs is None and getattr(tracker.scorer, "last_detail", None):
            for d in tracker.scorer.last_detail:
                print(f"      c={d.c:.2f} ν={d.nu:.2f} c·ν={d.contribution:.2f}"
                      f"  「{d.proposition[:40]}」", file=out)
    print("-" * 66, file=out)
    print(f"Final: λ1={tracker.lambda1:.2f}  λ2={tracker.lambda2:.2f}  "
          f"λ3={tracker.lambda3:.2f}  λ4={tracker.lambda4:.2f}   "
          f"(* = post merged into the previous one by consecutive-post compression)",
          file=out)
    return tracker


def _toy_example() -> None:
    """Worked example of Section 3.4: A*BCA*BC with q_m = 1 for A and 0 for B, C."""
    posts = [("A", "proposal"), ("B", "I agree."), ("C", "I agree."),
             ("A", "proposal"), ("B", "I agree."), ("C", "I agree.")]
    run_stream("toy example A*BCA*BC (q_m given explicitly)", posts,
               qs=[1, 0, 0, 1, 0, 0])


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Discussion-activeness models lambda1-lambda4")
    ap.add_argument("pattern_file", nargs="?", default=None,
                    help="discussion file in the '[m=1] Alice: ...' format "
                         "(omit to run a toy example without the NLI model)")
    ap.add_argument("--nli-model", default=HFNLIBackend.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None, help="model revision on the HuggingFace Hub")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: automatic)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print c(p) and nu(p) for each proposition")
    args = ap.parse_args(argv[1:])

    if args.pattern_file is None:
        _toy_example()
        return
    scorer = NLIContentScorer.zero_shot(model_name=args.nli_model,
                                        revision=args.revision, device=args.device)
    for name, posts in parse_pattern_file(args.pattern_file):
        run_stream(name, posts, scorer=scorer, verbose=args.verbose)


if __name__ == "__main__":
    main(sys.argv)
