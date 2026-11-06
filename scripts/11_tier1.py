"""Tier 1: hand-crafted features + LightGBM.

Also writes SHAP-style attributions (LightGBM pred_contrib) grouped by feature family,
which shows directly how much of the model's decision rests on length.

  python scripts/11_tier1.py [--no-emb] [--no-readability]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from prefbias.data import load_split, probs_frame
from prefbias.features import build_features, embedding_sims, feature_family
from prefbias.metrics import log_loss
from prefbias.paths import EVAL_SETS, PROC_DIR, run_dir


def features_for(name, use_emb, readability):
    cache = PROC_DIR / f"feats_{name}{'_emb' if use_emb else ''}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_split(name)
    sims = embedding_sims(df) if use_emb else None
    X = build_features(df, sims, readability)
    X.to_parquet(cache, index=False)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--no-readability", action="store_true")
    ap.add_argument("--run", default="tier1_lgbm")
    args = ap.parse_args()
    use_emb, rd = not args.no_emb, not args.no_readability

    train, val = load_split("train"), load_split("val")
    Xtr, Xva = features_for("train", use_emb, rd), features_for("val", use_emb, rd)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=42)
    model = lgb.train(params, lgb.Dataset(Xtr, train["label"]), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(Xva, val["label"])],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    d = run_dir(args.run)
    model.save_model(str(d / "model.txt"))

    for s in EVAL_SETS:
        df, X = load_split(s), features_for(s, use_emb, rd)
        P = model.predict(X, num_iteration=model.best_iteration)
        probs_frame(df["id"], P).to_parquet(d / f"preds_{s}.parquet", index=False)
        if s == "test":
            print("test log loss", log_loss(df["label"].to_numpy(), P))
            # attribution: mean |contribution| per feature, summed over classes, grouped by family
            contrib = model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)
            k = X.shape[1] + 1  # +1 bias column per class
            imp = sum(np.abs(contrib[:, c * k:(c + 1) * k - 1]).mean(0) for c in range(3))
            fi = pd.DataFrame({"feature": X.columns, "mean_abs_contrib": imp})
            fi["family"] = fi["feature"].map(feature_family)
            fi.sort_values("mean_abs_contrib", ascending=False).to_csv(d / "feature_attribution.csv", index=False)
            fam = (fi.groupby("family")["mean_abs_contrib"].sum() / fi["mean_abs_contrib"].sum()).round(4)
            (d / "family_share.json").write_text(json.dumps(fam.to_dict(), indent=2))
            print("attribution share by family:", fam.to_dict())
    (d / "params.json").write_text(json.dumps({**params, "best_iteration": model.best_iteration,
                                               "use_emb": use_emb, "readability": rd}))


if __name__ == "__main__":
    main()
