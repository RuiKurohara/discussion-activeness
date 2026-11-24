# -*- coding: utf-8 -*-
"""
Generate the discussion instances of the five patterns (P1-P5) with TinyTroupe.

This is the script used to generate the 50 instances in data/pattern_dataset/
(ten instances per pattern) with TinyTroupe 0.5.2 and OpenAI gpt-4.1-mini.
Apart from translated comments, output messages, and pattern descriptions,
and the output path, the generation logic is unchanged.

Conditions
  - All instances share the same topic:
      "How should we design an energy policy that balances stable
       energy supply with the realization of a decarbonized society?"
  - Every turn consists of exactly two sentences (propositions), including
    turns that only express agreement or acknowledgment.
  - Speaking orders (Table 1 of the paper):
      P1: AAAAAA (abstract content)
      P2: AAAAAA (concrete content)
      P3: ABCABC (B and C only agree or acknowledge)
      P4: AABBCC (no references to one another)
      P5: ABCABC (each speaker refers to the preceding speaker)
  - To keep the ten instances of a pattern from becoming too similar, each
    instance receives a different starting perspective at its first turn
    (ANGLES; ABSTRACT_FRAMES for P1). No further guidance is given afterwards.

Output
  data/pattern_dataset/P1/P1_01.txt ... P5/P5_10.txt

Usage
  python generate_pattern_dataset.py            # all patterns, ten instances each
  python generate_pattern_dataset.py P3         # the ten instances of P3 only
  python generate_pattern_dataset.py P3 5       # the fifth instance of P3 only

This calls the OpenAI API: a valid OPENAI_API_KEY is required and API usage
is charged. Because the LLM is sampled (temperature 1.5 in config.ini), a new
run produces different texts from the released ones.
"""

import os
import re
import sys
from pathlib import Path

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld

# ============================================================
# Compatibility shim: disable consolidate_episode_memories() so that the
# agents keep the whole (short) conversation in their episodic memory.
# ============================================================
try:
    def _no_consolidation(self, *args, **kwargs):
        return None
    TinyPerson.consolidate_episode_memories = _no_consolidation
    if hasattr(TinyPerson, "MAX_EPISODE_LENGTH"):
        TinyPerson.MAX_EPISODE_LENGTH = 10 ** 9
except Exception as _e:
    print(f"(warning: could not apply the consolidation shim: {_e})")

# ============================================================
# Basic settings
# ============================================================
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set the OPENAI_API_KEY environment variable first.")

TOPIC = ("How should we design an energy policy that balances stable "
         "energy supply with the realization of a decarbonized society?")

N_PROPOSITIONS = 2   # number of sentences (propositions) per turn, for every turn
N_RUNS = 10
MAX_ATTEMPTS = 3      # retries when the number of sentences is wrong
OUT_ROOT = Path(__file__).resolve().parent / "data" / "pattern_dataset"

# Big Five trait levels assigned according to the role of each agent
BIG5_DOMINANT = {"openness": "High", "conscientiousness": "High",
                 "extraversion": "High", "agreeableness": "Low",
                 "neuroticism": "Low"}
BIG5_PASSIVE = {"openness": "Low", "conscientiousness": "Low",
                "extraversion": "Low", "agreeableness": "High",
                "neuroticism": "High"}
BIG5_ENGAGED = {"openness": "High", "conscientiousness": "High",
                "extraversion": "High", "agreeableness": "High",
                "neuroticism": "Low"}

# Basic profile shared by all agents
BASE_PERSONA = {
    "Alice": {"age": 24, "nationality": "Japanese",
              "residence": "Tokyo, Japan", "occupation": "company employee"},
    "Bob":   {"age": 24, "nationality": "Japanese",
              "residence": "Tokyo, Japan", "occupation": "company employee"},
    "Carol": {"age": 24, "nationality": "Japanese",
              "residence": "Tokyo, Japan", "occupation": "company employee"},
}

# Starting perspectives for the first turn of the concrete patterns (P2-P5)
ANGLES = [
    "expanding offshore wind and solar capacity",
    "restarting or extending nuclear power plants",
    "grid-scale battery storage and smart grid upgrades",
    "demand-side management and dynamic electricity pricing",
    "a carbon border tax and industrial decarbonization incentives",
    "nationwide building energy-efficiency retrofits",
    "green hydrogen production and industrial use",
    "expanding regional grid interconnection and transmission capacity",
    "electric vehicle infrastructure and transportation electrification",
    "geothermal energy development and community heating networks",
]

# Starting perspectives for the first turn of the abstract pattern P1
# (no technology names)
ABSTRACT_FRAMES = [
    "the general importance of taking action",
    "the complexity of balancing multiple priorities",
    "the need for careful long-term planning",
    "the urgency of addressing the issue soon",
    "the difficulty of finding the right approach",
    "the importance of considering everyone's needs",
    "the challenge of making good decisions here",
    "the significance of this issue for the future",
    "the need to think about this holistically",
    "the importance of not rushing into anything",
]

# ============================================================
# Pattern definitions (speaking orders as in Table 1 of the paper)
# ============================================================
PATTERNS = {
    "P1": {
        "schedule": ["Alice"] * 6,
        "agents": {"Alice": {"big5": BIG5_DOMINANT, "style": "proposer_abstract"}},
        "desc": "One participant posts throughout; the content is abstract and "
                "lacks specificity",
    },
    "P2": {
        "schedule": ["Alice"] * 6,
        "agents": {"Alice": {"big5": BIG5_DOMINANT, "style": "proposer_specific"}},
        "desc": "One participant posts throughout and presents a concrete, "
                "topic-relevant proposal at every turn",
    },
    "P3": {
        "schedule": ["Alice", "Bob", "Carol", "Alice", "Bob", "Carol"],
        "agents": {
            "Alice": {"big5": BIG5_DOMINANT, "style": "proposer_specific"},
            "Bob":   {"big5": BIG5_PASSIVE, "style": "agree_only"},
            "Carol": {"big5": BIG5_PASSIVE, "style": "agree_only"},
        },
        "desc": "The speaking order is mutual, but B and C only express agreement "
                "or acknowledgment",
    },
    "P4": {
        "schedule": ["Alice", "Alice", "Bob", "Bob", "Carol", "Carol"],
        "agents": {
            "Alice": {"big5": BIG5_ENGAGED, "style": "proposer_specific_independent"},
            "Bob":   {"big5": BIG5_ENGAGED, "style": "proposer_specific_independent"},
            "Carol": {"big5": BIG5_ENGAGED, "style": "proposer_specific_independent"},
        },
        "desc": "All participants make concrete contributions but do not refer to "
                "one another",
    },
    "P5": {
        "schedule": ["Alice", "Bob", "Carol", "Alice", "Bob", "Carol"],
        "agents": {
            "Alice": {"big5": BIG5_ENGAGED, "style": "builder_crossref"},
            "Bob":   {"big5": BIG5_ENGAGED, "style": "builder_crossref"},
            "Carol": {"big5": BIG5_ENGAGED, "style": "builder_crossref"},
        },
        "desc": "Concrete contributions that build on the discussion by referring "
                "to the preceding speaker",
    },
}


# ============================================================
# Speaking style -> natural-language directive (all require exactly n sentences)
# ============================================================
def style_directive(style: str, n: int) -> str:
    if style == "proposer_abstract":
        return (f"You lead the discussion. Whenever you speak, state your own opinion "
                f"(not a reply) in EXACTLY {n} sentences. Keep the content vague and "
                f"generic: do NOT mention any specific technology, number, program, or "
                f"actionable policy detail. Just express general sentiment about the "
                f"importance or difficulty of the topic.")
    if style == "proposer_specific":
        return (f"You lead the discussion. Whenever you speak, state your own opinion "
                f"(not a reply) in EXACTLY {n} sentences, and include a new concrete, "
                f"actionable policy proposal that has not been mentioned before in this "
                f"conversation. Be specific: name technologies, mechanisms, or measures.")
    if style == "proposer_specific_independent":
        return (f"State your own opinion on the topic in EXACTLY {n} sentences, "
                f"including a new concrete, actionable policy proposal you have not "
                f"mentioned before. Be specific. Do NOT refer to or acknowledge other "
                f"participants or their statements: present your point as a fresh, "
                f"independent contribution.")
    if style == "builder_crossref":
        return (f"Reply directly to the previous speaker(s) by name, in EXACTLY {n} "
                f"sentences. Briefly acknowledge their point, then add a new concrete, "
                f"actionable policy proposal that has not been mentioned before. Be "
                f"specific.")
    if style == "agree_only":
        return (f"You NEVER propose anything new. Reply to the previous speaker with "
                f"EXACTLY {n} short sentences that only express agreement (for example, "
                f"'I agree.' followed by another short affirming remark). Do not add any "
                f"new information, reasons, proposals, or technical detail.")
    raise ValueError(f"unknown style: {style}")


def clear_agent_registry():
    """Clear TinyTroupe's agent registry so that agents with the same names can be
    created again for the next instance."""
    for attr in ("all_agents", "_all_agents"):
        reg = getattr(TinyPerson, attr, None)
        if isinstance(reg, dict):
            reg.clear()
            return
    for meth in ("clear_agents", "clear_all_agents"):
        fn = getattr(TinyPerson, meth, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def make_agent(name: str, cfg: dict) -> TinyPerson:
    agent = TinyPerson(name)
    for k, v in BASE_PERSONA[name].items():
        agent.define(k, v)
    for k, v in cfg["big5"].items():
        agent.define(k, v)
    agent.define("communication_directives",
                 style_directive(cfg["style"], N_PROPOSITIONS))
    return agent


def extract_talk(actions) -> str | None:
    contents = []
    for a in actions or []:
        act = a.get("action", a) if isinstance(a, dict) else None
        if act and act.get("type") == "TALK" and act.get("content"):
            contents.append(str(act["content"]).strip())
    return " ".join(contents) if contents else None


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: str) -> int:
    parts = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s]
    return len(parts)


def run_one(pattern_name: str, cfg: dict, run_idx: int, opening_hint: str) -> str:
    clear_agent_registry()
    agents = {name: make_agent(name, a) for name, a in cfg["agents"].items()}
    world = TinyWorld(f"Chat ({pattern_name}-{run_idx:02d})", list(agents.values()))
    world.make_everyone_accessible()

    for a in agents.values():
        a.listen(f"Discussion topic: {TOPIC}")

    schedule = cfg["schedule"]
    utterances = []
    for t, speaker_name in enumerate(schedule, start=1):
        speaker = agents[speaker_name]
        moderator_msg = ("[Moderator] It is now your turn. "
                          "Please post your message to the chat room.")
        if t == 1:
            moderator_msg += f" To start, focus your point on {opening_hint}."
        speaker.listen(moderator_msg)

        content = None
        for _attempt in range(MAX_ATTEMPTS):
            actions = speaker.act(return_actions=True)
            content = extract_talk(actions)
            if content and count_sentences(content) == N_PROPOSITIONS:
                break
            if content is None:
                speaker.listen("[Moderator] Please speak now, following your "
                               "communication directives.")
            else:
                speaker.listen(f"[Moderator] Your message must contain EXACTLY "
                               f"{N_PROPOSITIONS} sentences (it currently has "
                               f"{count_sentences(content)}). Please rewrite it.")
        if not content:
            print(f"(warning: {pattern_name}_{run_idx:02d} m={t} {speaker_name} "
                  f"did not speak)")
            content = "(no response)"

        utterances.append((speaker_name, content))
        for name, other in agents.items():
            if name != speaker_name:
                other.listen(f"{speaker_name} said: {content}")

    lines = [f"==== {pattern_name}_{run_idx:02d}: {cfg['desc']} ====",
             f"topic: {TOPIC}",
             f"opening_hint: {opening_hint}", ""]
    for i, (speaker, text) in enumerate(utterances, start=1):
        n = count_sentences(text)
        mark = "" if n == N_PROPOSITIONS else f"  [!sentences={n}]"
        lines.append(f"[m={i}] {speaker}: {text}{mark}")
    return "\n".join(lines) + "\n"


def main(argv):
    target_patterns = ([argv[1]] if len(argv) > 1 and argv[1] in PATTERNS
                        else list(PATTERNS.keys()))
    only_run = int(argv[2]) if len(argv) > 2 else None

    for pname in target_patterns:
        cfg = PATTERNS[pname]
        frames = ABSTRACT_FRAMES if pname == "P1" else ANGLES
        out_dir = OUT_ROOT / pname
        out_dir.mkdir(parents=True, exist_ok=True)
        run_range = [only_run] if only_run else range(1, N_RUNS + 1)
        for run_idx in run_range:
            hint = frames[(run_idx - 1) % len(frames)]
            print(f"\n=== generating {pname}_{run_idx:02d} (hint: {hint}) ===")
            text = run_one(pname, cfg, run_idx, hint)
            out_path = out_dir / f"{pname}_{run_idx:02d}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(text)
            print(f"saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv)
