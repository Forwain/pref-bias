"""Exploratory analysis: how much position and length bias is already in the human labels.

Outputs outputs/eda/{eda.json, *.png}. Uses the train split only (test stays untouched).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prefbias.data import load_split
from prefbias.paths import run_dir


def main():
    out = run_dir("eda")
    df = load_split("train")
    nt = df[df["label"] != 2]
    longer_a = nt["len_a"] > nt["len_b"]
    stats = {
        "rows": len(df),
        "label_dist(A,B,tie)": df["label"].value_counts(normalize=True).sort_index().round(4).tolist(),
        "P(A wins)-P(B wins)": round(float((df.label == 0).mean() - (df.label == 1).mean()), 4),
        "P(longer wins | not tie)": round(float(np.mean(np.where(longer_a, nt.label == 0, nt.label == 1))), 4),
        "multi_turn_share": round(float((df.n_turns > 1).mean()), 4),
        "empty_response_share": round(float(((df.len_a == 0) | (df.len_b == 0)).mean()), 4),
        "resp_chars_median": float(np.median(np.r_[df.len_a, df.len_b])),
        "resp_chars_p95": float(np.percentile(np.r_[df.len_a, df.len_b], 95)),
        "n_models": int(pd.concat([df.model_a, df.model_b]).nunique()),
        "top_models": pd.concat([df.model_a, df.model_b]).value_counts().head(15).to_dict(),
    }
    # tie rate vs. how similar the two lengths are
    absr = df["log_len_ratio"].abs()
    stats["tie_rate_by_abs_len_ratio"] = (
        df.groupby(pd.qcut(absr, 5, duplicates="drop"), observed=True)["label"]
        .apply(lambda s: round(float((s == 2).mean()), 4)).astype(str).to_dict()
    )
    (out / "eda.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))

    # Figure 1: win rate of A vs. log length ratio
    bins = pd.qcut(df["log_len_ratio"], 20, duplicates="drop")
    g = df.groupby(bins, observed=True)["label"]
    x = df.groupby(bins, observed=True)["log_len_ratio"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, name in enumerate(["A wins", "B wins", "tie"]):
        ax.plot(x, g.apply(lambda s: (s == k).mean()), marker="o", label=name)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("log(len A / len B)"); ax.set_ylabel("share of human votes")
    ax.set_title("Human preference vs. length difference (train split)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "win_rate_vs_length.png", dpi=150)

    # Figure 2: response length distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(np.r_[df.len_a, df.len_b] + 1), bins=60)
    ax.set_xlabel("log10(response characters + 1)"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(out / "length_hist.png", dpi=150)


if __name__ == "__main__":
    main()
