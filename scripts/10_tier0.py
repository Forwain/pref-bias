"""Tier 0 heuristics: no model, just a rule.

tier0_position: always predict the train-set class prior (it encodes "slot A wins a bit more").
tier0_length:   lookup table of class frequencies per bin of log(len_A / len_B), fitted on
                train + swapped train so it uses length only, not position (~60 numbers).
"""
import json

import numpy as np

from prefbias.data import load_split, probs_frame, swap_ab
from prefbias.paths import EVAL_SETS, run_dir


def main(n_bins=20, alpha=1.0):
    train = load_split("train")
    evals = {s: load_split(s) for s in EVAL_SETS}

    # position rule
    prior = np.bincount(train["label"], minlength=3) / len(train)
    d = run_dir("tier0_position")
    for s, df in evals.items():
        probs_frame(df["id"], np.tile(prior, (len(df), 1))).to_parquet(d / f"preds_{s}.parquet", index=False)
    (d / "params.json").write_text(json.dumps({"prior": prior.tolist()}))

    # length rule (symmetrised)
    sym = np.r_[train["log_len_ratio"], -train["log_len_ratio"]]
    lab = np.r_[train["label"], swap_ab(train)["label"]]
    edges = np.unique(np.quantile(sym, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(sym, edges[1:-1])
    table = np.zeros((len(edges) - 1, 3)) + alpha
    np.add.at(table, (b, lab), 1)
    table /= table.sum(1, keepdims=True)
    d = run_dir("tier0_length")
    for s, df in evals.items():
        P = table[np.digitize(df["log_len_ratio"], edges[1:-1])]
        probs_frame(df["id"], P).to_parquet(d / f"preds_{s}.parquet", index=False)
    (d / "params.json").write_text(json.dumps({"edges": edges[1:-1].tolist(), "table": table.tolist()}))
    print("tier0 done")


if __name__ == "__main__":
    main()
