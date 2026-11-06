"""Loading, parsing and splitting the Kaggle "LLM Classification Finetuning" data.

train.csv columns: id, model_a, model_b, prompt, response_a, response_b,
winner_model_a, winner_model_b, winner_tie.
prompt / response_* are JSON-encoded lists of strings (one entry per conversation turn).
"""
import hashlib
import json

import numpy as np
import pandas as pd

from .paths import PROC_DIR, RAW_DIR

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
LABEL_NAMES = ["model_a", "model_b", "tie"]  # class index 0, 1, 2


def _clean(t) -> str:
    if not isinstance(t, str):
        return ""
    # Some rows contain lone UTF-16 surrogates that break parquet / tokenizers.
    return t.encode("utf-8", "replace").decode("utf-8")


def parse_turns(s) -> list:
    """Parse a JSON list of turns. Null turns become empty strings."""
    if not isinstance(s, str):
        return []
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return [_clean(s)]
    if isinstance(v, str) or v is None:
        v = [v]
    return [_clean(t) for t in v]


def join_turns(turns: list) -> str:
    return "\n\n".join(t for t in turns if t)


def load_raw(name: str = "train.csv") -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / name)
    for col in ["prompt", "response_a", "response_b"]:
        df[col + "_turns"] = df[col].map(parse_turns)
        df[col + "_text"] = df[col + "_turns"].map(join_turns)
    df = df.rename(columns={
        "response_a_text": "resp_a",
        "response_b_text": "resp_b",
        "response_a_turns": "turns_a",
        "response_b_turns": "turns_b",
        "prompt_turns": "turns_prompt",
    })
    df = df.drop(columns=["prompt", "response_a", "response_b"])
    if all(c in df.columns for c in LABEL_COLS):
        df["label"] = df[LABEL_COLS].to_numpy().argmax(1)
    df["n_turns"] = df["turns_prompt"].map(len)
    add_lengths(df)
    return df


def add_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df["len_a"] = df["resp_a"].str.len()
    df["len_b"] = df["resp_b"].str.len()
    df["log_len_ratio"] = np.log((df["len_a"] + 1) / (df["len_b"] + 1))
    return df


def split_of(prompt_text: str) -> str:
    """Deterministic 80/10/10 split grouped by prompt, so one prompt never spans two splits."""
    h = int(hashlib.md5(prompt_text.strip().lower().encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    """Swap response A and B (and their labels): the position-bias probe and the debias augmentation."""
    s = df.copy()
    pairs = [("resp_a", "resp_b"), ("turns_a", "turns_b"), ("len_a", "len_b")]
    if "model_a" in s.columns:
        pairs.append(("model_a", "model_b"))
    for a, b in pairs:
        s[a], s[b] = df[b].to_numpy(), df[a].to_numpy()
    s["log_len_ratio"] = -df["log_len_ratio"]
    if "label" in s.columns:
        s["label"] = df["label"].map({0: 1, 1: 0, 2: 2})
    s["swapped"] = True
    return s


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / f"{name}.parquet")


def probs_frame(ids, P: np.ndarray) -> pd.DataFrame:
    """Standard prediction format shared by every tier."""
    return pd.DataFrame({"id": np.asarray(ids), "p_a": P[:, 0], "p_b": P[:, 1], "p_tie": P[:, 2]})
