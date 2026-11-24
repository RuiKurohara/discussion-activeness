# Discussion Activeness Models: Post Mutuality and Content Quality

Code and data for the paper

> Rui Kurohara, Wen Gu, Takuto Sakuma, and Shohei Kato.
> **Modeling Discussion Activeness in Online Forums Based on Post Mutuality and Content Quality.**
> In *Proceedings of the International Conference on Agents and Artificial Intelligence (ICAART)*, to appear.

The repository contains the implementation of the four activeness models compared in the paper,
the script that generated the 50 LLM-agent discussion instances (five patterns × ten instances),
the generated instances themselves, and the scores reported in Table 1 and Figures 3 and 4.

## Contents

| Path | Description |
|---|---|
| `activity_models.py` | Streaming implementation of the activeness models and the zero-shot NLI content scorer (Section 3) |
| `evaluate_patterns.py` | Computes the models for all 50 instances (Table 1) |
| `plot_pattern_comparison.py` | Figure 3 (box plots of the final scores) |
| `plot_turn_progression.py` | Figure 4 (trajectories over speaking turns) |
| `generate_pattern_dataset.py` | Generates the discussion instances with TinyTroupe and GPT-4.1-mini (Section 4.1) |
| `config.ini` | TinyTroupe settings used for the generation |
| `data/pattern_dataset/` | The 50 generated discussion instances (`P1/P1_01.txt` … `P5/P5_10.txt`) |
| `results/` | Outputs of `evaluate_patterns.py` and the plotting scripts |
| `tests/test_worked_examples.py` | Checks the worked examples of Sections 3.2 and 3.4 (no model needed) |

## Models

| Code | Paper | Definition |
|---|---|---|
| `lambda1` | λ_baseline (Eq. 8) | number of posts \|M(t)\| |
| `lambda2` | λ_mutuality (Eq. 3) | (1 − G(t)) Σ_{b_m ∈ M'(t)} I_m |
| `lambda3` | λ_content (Eq. 6) | Σ_{x_m ∈ M(t)} q_m |
| `lambda4` | λ_mix (Eq. 7) | (1 − G(t)) Σ_{b_m ∈ M'(t)} I_m q_m |

* M'(t) merges consecutive posts by the same speaker into one post. I_m is the number of posts by
  other speakers between two successive contributions of a speaker on M'(t) (1 for a first contribution).
* G(t) is the Gini coefficient of the uncompressed post counts per speaker.
* q_m = Σ_p c(p) ν(p) over the sentences (propositions) p of a post.
  c(p) is the specificity and ν(p) = 1 − max_{p'} P(Rephrase | p, p') is the novelty with respect to all
  earlier propositions. For a compressed post, q_m is the sum over the merged posts.

`ActivityTracker.add_post(speaker, text)` updates all four models each time a post arrives.

### Zero-shot NLI content scorer

Both c(p) and P(Rephrase | p, p') are computed with the NLI model
[`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
without additional training.

* **c(p)**: with p as the premise, the entailment probabilities of the two hypotheses below are
  normalized against each other, c(p) = P_spec / (P_spec + P_gen).
  * `"This makes a concrete proposal, claim, or new piece of information."` (P_spec)
  * `"This is just agreement or acknowledgment, with no new information."` (P_gen)
* **P(Rephrase | p, p')** ≈ P(p ⇒ p') × P(p' ⇒ p) (bidirectional entailment).
* Posts are split into sentences at `.`, `!`, and `?`; each sentence is one proposition.

## Quick start

Python 3.12 was used.

```bash
pip install -r requirements.txt

python tests/test_worked_examples.py     # worked examples of Sections 3.2 and 3.4
python activity_models.py                # toy example with explicit q_m (no model download)
python activity_models.py data/pattern_dataset/P3/P3_01.txt -v   # one instance with the NLI scorer
```

## Reproducing Table 1 and Figures 3 and 4

```bash
python evaluate_patterns.py --device cpu --revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c
python plot_pattern_comparison.py
python plot_turn_progression.py
```

`evaluate_patterns.py` writes

* `results/activity_log.txt`: values of all models at every turn, with c(p) and ν(p) of every proposition,
* `results/summary_by_run.csv`: final values of each instance,
* `results/summary_by_pattern.csv`: mean and **population** standard deviation (ddof = 0) over the ten
  instances of each pattern, as reported in Table 1.

The `--revision` option pins the model files on the Hugging Face Hub to the revision used for the
verification below. The proposition history used for ν(p) is reset before every instance.

### Verification

The files in `results/` were produced with the command above on a CPU (Python 3.12.10, torch 2.11.0,
transformers 5.19.0). They are identical to the outputs of the original experiment: all 50 final
values (to two decimal places), all 300 per-turn rows, and the c(p) and ν(p) values of all 600
propositions. `results/summary_by_pattern.csv` therefore reproduces Table 1 exactly.

Running the same code on a CUDA GPU changed some instance-level values by ±0.01 because of
floating-point differences; the pattern-level statistics then differed from Table 1 by at most 0.01
(e.g., the standard deviation of λ_content for P3 was 0.54 instead of 0.55). Use `--device cpu` to
reproduce Table 1 exactly.

## Regenerating the discussion instances

The released instances were generated with
[TinyTroupe](https://github.com/microsoft/TinyTroupe) 0.5.2
(commit `43d951eea015aea09972dc95537c9bbbc74ba207`) and OpenAI `gpt-4.1-mini`, using the settings in
`config.ini` (temperature 1.5).

```bash
pip install git+https://github.com/microsoft/TinyTroupe.git@43d951eea015aea09972dc95537c9bbbc74ba207
export OPENAI_API_KEY=...                    # API usage is charged
python generate_pattern_dataset.py           # all patterns, ten instances each
python generate_pattern_dataset.py P3        # the ten instances of P3 only
python generate_pattern_dataset.py P3 5      # the fifth instance of P3 only
```

Run the script from this folder so that TinyTroupe reads `config.ini`. Because the LLM output is
sampled, new runs produce texts that differ from the released ones; the released texts are the ones
analyzed in the paper. Note that the script writes to `data/pattern_dataset/` and overwrites the
released files.

Generation conditions (see the script for the exact prompts):

* Topic (all instances): *"How should we design an energy policy that balances stable energy supply
  with the realization of a decarbonized society?"*
* Six turns per instance; every turn must contain exactly two sentences, including agreement-only turns.
  If a turn does not have two sentences, the agent is asked to rewrite it (up to three attempts).
* Agents Alice, Bob, and Carol correspond to A, B, and C in the speaking-order notation.
  All agents share the same basic profile (age 24, Japanese, living in Tokyo, company employee) and are
  given Big Five trait levels according to their roles. The speaking style of each pattern is specified
  by a natural-language directive (`style_directive` in the script).
* At the first turn only, each instance receives one of ten starting perspectives (`ANGLES`; for P1,
  `ABSTRACT_FRAMES`, which do not name technologies). No further guidance is given afterwards.

| Pattern | Speaking order | Description |
|---|---|---|
| P1 | A\*A\*A\*A\*A\*A\* | One participant; abstract statements without concrete proposals |
| P2 | A\*A\*A\*A\*A\*A\* | One participant; a concrete, topic-relevant proposal at every turn |
| P3 | A\*BCA\*BC | Mutual speaking order, but B and C only agree or acknowledge |
| P4 | A\*A\*B\*B\*C\*C\* | Concrete contributions without references to one another |
| P5 | A\*B\*C\*A\*B\*C\* | Concrete contributions that refer to the preceding speaker |

## Data format

```
==== P3_01: <pattern description> ====
topic: <discussion topic>
opening_hint: <starting perspective given at the first turn>

[m=1] Alice: <two sentences>
[m=2] Bob: <two sentences>
...
[m=6] Carol: <two sentences>
```

Notes:

* The pattern descriptions in the header lines were originally written in Japanese and have been
  translated into English for this release. The utterances are unchanged.
* In some P5 instances, the first turn starts with an agreement phrase such as "I agree that ..."
  although there is no preceding speaker (e.g., `P5/P5_07.txt`, m=1).
* A post marked `[!sentences=N]` would indicate a turn that did not have two sentences after all
  retries; none of the 50 released instances has this mark.

## License

The code and data in this repository are released under the MIT License (see `LICENSE`).
TinyTroupe is a separate project by Microsoft, distributed under its own license.

## Citation

If you use this code or data, please cite the paper above.
