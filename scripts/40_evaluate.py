"""Compute every metric for every run and write the results table.

For each run directory under outputs/ that has preds_test.parquet:
  predictive: log loss, accuracy, ECE on test
  position:   consistency and position bias (needs preds_test_swap)
  length:     slope of P(A)/(P(A)+P(B)) on log length ratio vs. the human slope
  verbosity:  gain / success / flip under the restate and filler probes
  self-pref:  for --family (default qwen, the judge family; for other tiers it is a control)
A derived "<run>+swapTTA" row averages the original and un-swapped predictions
(inference-time debiasing, no retraining). 95% bootstrap CIs on the key metrics.

  python scripts/40_evaluate.py [--runs tier0_length tier1_lgbm ...] [--B 200]
"""
import argparse
import json

import numpy as np
import pandas as pd

from prefbias import metrics as M
from prefbias.data import load_split
from prefbias.paths import OUT_DIR

VERBOSE = ["test_verbose_restate", "test_verbose_filler"]


def load_preds(run, s, ids):
    f = OUT_DIR / run / f"preds_{s}.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f).set_index("id")
    if not set(ids) <= set(p.index):
        return None  # e.g. a --limit run that covers only part of the set
    return p.loc[ids, ["p_a", "p_b", "p_tie"]].to_numpy()


def evaluate(name, P, P_swap, P_verb, test, family, B):
    y, x = test["label"].to_numpy(), test["log_len_ratio"].to_numpy()
    n = len(y)
    r = {"run": name, "n": n,
         "log_loss": M.log_loss(y, P), "acc": M.accuracy(y, P), "ece": M.ece(y, P)}
    r["log_loss_ci"] = M.bootstrap_ci(lambda i: M.log_loss(y[i], P[i]), n, B)
    r.update(M.length_sensitivity(x, y, P))
    r["len_slope_excess_ci"] = M.bootstrap_ci(
        lambda i: M.length_sensitivity(x[i], y[i], P[i])["len_slope_excess"], n, B)
    if P_swap is not None:
        r["pos_consistency"] = M.position_consistency(P, P_swap)
        r["pos_bias"] = M.position_bias(P, P_swap)
        r["pos_consistency_ci"] = M.bootstrap_ci(lambda i: M.position_consistency(P[i], P_swap[i]), n, B)
        r["log_loss_swap"] = M.log_loss(y, M.unswap(P_swap))
    for s, (Pv, side) in P_verb.items():
        for k, v in M.verbosity_attack(P, Pv, side).items():
            r[f"{k}_{s.split('_')[-1]}"] = v
    r.update(M.self_preference(P, y, test["model_a"], test["model_b"], family))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--family", default="qwen",
                    help="model family for the self-preference metric (the Tier 2 judge is Qwen)")
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    test = load_split("test")
    ids = test["id"].to_numpy()
    verb_sides = {s: load_split(s).set_index("id").loc[ids, "inflated_side"].to_numpy() for s in VERBOSE}
    runs = args.runs or sorted(d.name for d in OUT_DIR.iterdir() if (d / "preds_test.parquet").exists())

    y = test["label"].to_numpy()
    nt = y != 2
    rows = [{"run": "human_labels", "n": len(y), "pos_bias": M.human_position_bias(y),
             "len_slope_model": M.soft_logistic_slope(test["log_len_ratio"].to_numpy()[nt], (y[nt] == 0).astype(float))}]
    for run in runs:
        P = load_preds(run, "test", ids)
        if P is None:
            print(f"skip {run}: no full test predictions"); continue
        P_swap = load_preds(run, "test_swap", ids)
        P_verb = {s: (Pv, verb_sides[s]) for s in VERBOSE if (Pv := load_preds(run, s, ids)) is not None}
        rows.append(evaluate(run, P, P_swap, P_verb, test, args.family, args.B))
        if P_swap is not None and not run.startswith("tier0"):
            P_tta = 0.5 * (P + M.unswap(P_swap))
            # the TTA prediction for the swapped pair is the original TTA prediction with A/B relabelled
            rows.append(evaluate(run + "+swapTTA", P_tta, M.unswap(P_tta), {}, test, args.family, args.B))
        print(f"done {run}")

    res = pd.DataFrame(rows)
    out = OUT_DIR / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    cols = [c for c in ["run", "log_loss", "acc", "ece", "pos_consistency", "pos_bias", "len_slope_model",
                        "len_slope_excess", "verb_gain_restate", "verb_flip_restate", "verb_gain_filler",
                        "verb_flip_filler", "self_pref_excess"] if c in res.columns]
    (out / "results.md").write_text(res[cols].round(4).to_markdown(index=False))
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
