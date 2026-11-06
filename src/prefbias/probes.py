"""Counterfactual probes: change one quality-irrelevant factor and watch the prediction.

Position probe: swap A and B (data.swap_ab).
Verbosity probe: pad one response with text that adds no information, in two styles:
  - "restate": append a recap that repeats the response's own opening sentences
    (a rule-based version of the repetitive-list attack in Zheng et al., 2023);
  - "filler": append a generic, content-free closing paragraph.
"""
import re

import numpy as np
import pandas as pd

from .data import add_lengths, join_turns

FILLERS = [
    "I hope this explanation is helpful. Please keep in mind that there are many ways to look at "
    "this question, and the best approach can depend on your specific situation and goals. "
    "If you have any further questions or would like more details on any part of this answer, "
    "feel free to ask and I will be happy to help.",
    "Overall, this is an interesting topic with several aspects worth considering. It is always a good "
    "idea to think carefully about the details and to consult additional resources if needed. "
    "Let me know if there is anything else you would like me to clarify or expand on.",
    "To conclude, the points above summarize the main considerations. Every situation is a little "
    "different, so it can be useful to reflect on how these ideas apply to your own case. "
    "Please do not hesitate to reach out if you need any further assistance.",
]

_SENT = re.compile(r"(?<=[.!?])\s+")


def _restate(text: str) -> str:
    sents = [s for s in _SENT.split(text.strip()) if s]
    if not sents:
        return text
    recap = " ".join(sents[:3])
    return f"{text}\n\nTo recap the key points: {recap}"


def _filler(text: str, rng) -> str:
    return f"{text}\n\n{FILLERS[rng.integers(len(FILLERS))]}"


def inflate_turns(turns: list, mode: str, rng) -> list:
    """Pad the last non-empty assistant turn."""
    turns = list(turns)
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]:
            turns[i] = _restate(turns[i]) if mode == "restate" else _filler(turns[i], rng)
            break
    return turns


def make_verbose_probe(df: pd.DataFrame, mode: str, seed: int = 0) -> pd.DataFrame:
    """Pad a randomly chosen side of every pair. Column `inflated_side` records which (0=A, 1=B)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    side = rng.integers(0, 2, len(df))
    new_a, new_b = [], []
    for ta, tb, s in zip(df["turns_a"], df["turns_b"], side):
        if s == 0:
            new_a.append(inflate_turns(ta, mode, rng)); new_b.append(list(tb))
        else:
            new_a.append(list(ta)); new_b.append(inflate_turns(tb, mode, rng))
    out["turns_a"], out["turns_b"] = new_a, new_b
    out["resp_a"] = out["turns_a"].map(join_turns)
    out["resp_b"] = out["turns_b"].map(join_turns)
    out["inflated_side"] = side
    out["probe"] = f"verbose_{mode}"
    return add_lengths(out)
