"""Hand-crafted features for Tier 1 (LightGBM).

Each response gets surface features; the model sees A, B and A-B for each.
Feature families (used to attribute SHAP mass): length, format, style, semantic.
"""
import re

import numpy as np
import pandas as pd

_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M)
_HEADER = re.compile(r"^\s*#{1,6}\s", re.M)
_BOLD = re.compile(r"\*\*[^*]+\*\*")
_URL = re.compile(r"https?://")
_REFUSAL = re.compile(r"\b(I'?m sorry|I cannot|I can'?t|I am unable|As an AI|as a language model)\b", re.I)
_WORD = re.compile(r"\w+")

FAMILY = {
    "n_chars": "length", "n_words": "length", "n_lines": "length", "n_paragraphs": "length",
    "n_sents": "length",
    "n_bullets": "format", "n_headers": "format", "n_bold": "format", "n_code": "format", "n_urls": "format",
    "avg_word_len": "style", "uniq_ratio": "style", "n_question": "style", "n_exclaim": "style",
    "refusal": "style", "empty": "style", "flesch": "style",
    "sim_prompt": "semantic",
}


def response_features(text: str, readability: bool = True) -> dict:
    words = _WORD.findall(text)
    f = {
        "n_chars": len(text),
        "n_words": len(words),
        "n_lines": text.count("\n") + 1 if text else 0,
        "n_paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "n_sents": len(re.findall(r"[.!?](\s|$)", text)),
        "n_bullets": len(_BULLET.findall(text)),
        "n_headers": len(_HEADER.findall(text)),
        "n_bold": len(_BOLD.findall(text)),
        "n_code": text.count("```") // 2,
        "n_urls": len(_URL.findall(text)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "uniq_ratio": len(set(w.lower() for w in words)) / len(words) if words else 0.0,
        "n_question": text.count("?"),
        "n_exclaim": text.count("!"),
        "refusal": int(bool(_REFUSAL.search(text[:600]))),
        "empty": int(len(text.strip()) == 0),
    }
    if readability:
        import textstat
        f["flesch"] = textstat.flesch_reading_ease(text[:5000]) if words else 0.0
    return f


def build_features(df: pd.DataFrame, sims: dict = None, readability: bool = True) -> pd.DataFrame:
    """sims: optional {"sim_prompt_a": array, "sim_prompt_b": array, "sim_ab": array}."""
    fa = pd.DataFrame([response_features(t, readability) for t in df["resp_a"]])
    fb = pd.DataFrame([response_features(t, readability) for t in df["resp_b"]])
    if sims is not None:
        fa["sim_prompt"] = sims["sim_prompt_a"]
        fb["sim_prompt"] = sims["sim_prompt_b"]
    X = pd.DataFrame(index=range(len(df)))
    for c in fa.columns:
        X[f"a_{c}"] = fa[c].to_numpy()
        X[f"b_{c}"] = fb[c].to_numpy()
        X[f"d_{c}"] = fa[c].to_numpy() - fb[c].to_numpy()
    X["d_log_chars"] = np.log1p(fa["n_chars"]) - np.log1p(fb["n_chars"])
    X["prompt_chars"] = df["prompt_text"].str.len().to_numpy()
    X["n_turns"] = df["n_turns"].to_numpy()
    if sims is not None:
        X["sim_ab"] = sims["sim_ab"]
    return X


def feature_family(col: str) -> str:
    if col == "d_log_chars":
        return "length"
    if col in ("prompt_chars", "n_turns"):
        return "prompt"
    if col == "sim_ab":
        return "semantic"
    return FAMILY.get(col[2:], "other")


def embedding_sims(df: pd.DataFrame, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256):
    """Cosine similarities between prompt and each response, and between the two responses."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    enc = lambda s: m.encode([t[:4000] for t in s], batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=True)
    ep, ea, eb = enc(df["prompt_text"].tolist()), enc(df["resp_a"].tolist()), enc(df["resp_b"].tolist())
    return {"sim_prompt_a": (ep * ea).sum(1), "sim_prompt_b": (ep * eb).sum(1), "sim_ab": (ea * eb).sum(1)}
