"""Surface-cue explainability: how much of each judge's decisions can shallow features reproduce?

For every run with val + test predictions, fit a "mimic" on the val split that predicts the judge's
own output from surface features only (length, style, format; no embeddings, no semantics), then
score it on test. A high score means the judge's decisions are mostly a function of surface cues.

  python scripts/41_surface_explain.py       -> outputs/results/surface_explain.{csv,md}

Targets (A-vs-B preference; ties left to the per-class outputs):
  s = log P(A) - log P(B)    R^2 of the mimic on test. Invariant to the judge's temperature.
  d = 1[s > 0]               agreement of a mimic classifier with the judge's decision, and AUC.
Reference row "human_labels": the same classifier fitted to human votes (non-tie rows).
Feature subsets: all surface / length only (d_log_chars) / one family at a time.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

from prefbias.data import load_split
from prefbias.features import feature_family
from prefbias.paths import OUT_DIR, PROC_DIR, run_dir

REG = dict(objective="regression", n_estimators=400, learning_rate=0.03, num_leaves=15,
           min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
CLF = {**REG, "objective": "binary"}


def preference(run, split):
    P = pd.read_parquet(OUT_DIR / run / f"preds_{split}.parquet")
    pa, pb = np.clip(P["p_a"].to_numpy(), 1e-6, 1), np.clip(P["p_b"].to_numpy(), 1e-6, 1)
    return P["id"].to_numpy(), np.log(pa) - np.log(pb)


def fit_scores(Xv, yv, Xt, yt, continuous):
    out = {}
    if continuous:
        m = lgb.LGBMRegressor(**REG).fit(Xv, yv)
        out["r2"] = r2_score(yt, m.predict(Xt))
        out["r2_linear"] = r2_score(yt, LinearRegression().fit(Xv, yv).predict(Xt))
        dv, dt = (yv > 0).astype(int), (yt > 0).astype(int)
    else:
        dv, dt = yv, yt
    c = lgb.LGBMClassifier(**CLF).fit(Xv, dv)
    p = c.predict_proba(Xt)[:, 1]
    out["agree"] = float(np.mean((p > 0.5) == dt))
    out["auc"] = roc_auc_score(dt, p) if 0 < dt.mean() < 1 else np.nan
    out["majority"] = float(max(dt.mean(), 1 - dt.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

    Fv, Ft = pd.read_parquet(PROC_DIR / "feats_val.parquet"), pd.read_parquet(PROC_DIR / "feats_test.parquet")
    cols = [c for c in Fv.columns if feature_family(c) in ("length", "style", "format")]
    fam = {c: feature_family(c) for c in cols}
    subsets = {"all surface": cols, "length only": ["d_log_chars"]}
    for f in ("length", "style", "format"):
        subsets[f"{f} family"] = [c for c in cols if fam[c] == f]

    val, test = load_split("val"), load_split("test")
    runs = args.runs or sorted(p.name for p in OUT_DIR.iterdir()
                               if (p / "preds_val.parquet").exists() and (p / "preds_test.parquet").exists())
    rows = []

    # human reference: non-tie votes
    mv, mt = val["label"].to_numpy() != 2, test["label"].to_numpy() != 2
    for sname, sc in subsets.items():
        r = fit_scores(Fv.loc[mv, sc], (val["label"][mv] == 0).astype(int).to_numpy(),
                       Ft.loc[mt, sc], (test["label"][mt] == 0).astype(int).to_numpy(), continuous=False)
        rows.append({"run": "human_labels", "features": sname, **r})

    for run in runs:
        iv, sv = preference(run, "val")
        it, st = preference(run, "test")
        if len(iv) != len(val) or len(it) != len(test):
            print("skip (partial run):", run); continue
        assert (iv == val["id"].to_numpy()).all() and (it == test["id"].to_numpy()).all(), run
        if np.std(st) < 1e-6:
            print("skip (constant):", run); continue
        for sname, sc in subsets.items():
            r = fit_scores(Fv[sc], sv, Ft[sc], st, continuous=True)
            rows.append({"run": run, "features": sname, **r})
        print(run, "done", flush=True)

    tab = pd.DataFrame(rows)
    out = run_dir("results")
    tab.to_csv(out / "surface_explain.csv", index=False)
    wide = tab.pivot(index="run", columns="features", values="r2")
    agree = tab[tab.features == "all surface"].set_index("run")[["agree", "auc", "majority", "r2_linear"]]
    md = ["## R^2 of surface-feature mimic on judge preference log P(A)/P(B) (test)\n",
          wide.to_markdown(floatfmt=".3f"),
          "\n\n## Decision agreement of the all-surface mimic (test)\n", agree.to_markdown(floatfmt=".3f")]
    (out / "surface_explain.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
