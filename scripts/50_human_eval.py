"""Human evaluation and error-analysis sheets.

  python scripts/50_human_eval.py sample --n 200
      -> outputs/human_eval/annotation_sheet.csv  (Arena label hidden; two annotator columns)
  python scripts/50_human_eval.py score
      -> agreement of each annotator with the Arena vote, Cohen's kappa between annotators
  python scripts/50_human_eval.py errors --run tier3_deberta --n 50
      -> outputs/<run>/error_samples.csv: swap-inconsistent pairs and confident mistakes,
         with an empty `category` column for manual coding
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from prefbias.data import load_split
from prefbias.metrics import unswap
from prefbias.paths import OUT_DIR, run_dir

LETTER = {0: "A", 1: "B", 2: "C"}


def sample(n):
    test = load_split("test")
    # stratify by label so ties are represented
    s = test.groupby("label", group_keys=False).apply(lambda g: g.sample(n // 3 + 1, random_state=0)).head(n)
    s = s.sample(frac=1, random_state=1)
    d = run_dir("human_eval")
    sheet = s[["id", "prompt_text", "resp_a", "resp_b"]].copy()
    sheet["annotator_1"] = ""  # A / B / C (tie)
    sheet["annotator_2"] = ""
    sheet["notes"] = ""
    sheet.to_csv(d / "annotation_sheet.csv", index=False)
    s[["id", "label"]].to_csv(d / "answer_key.csv", index=False)  # do not share with annotators
    print(f"wrote {len(sheet)} rows to {d / 'annotation_sheet.csv'}")


def score():
    d = OUT_DIR / "human_eval"
    sheet = pd.read_csv(d / "annotation_sheet.csv").merge(pd.read_csv(d / "answer_key.csv"), on="id")
    sheet = sheet.dropna(subset=["annotator_1", "annotator_2"])
    arena = sheet["label"].map(LETTER)
    a1, a2 = sheet["annotator_1"].str.strip().str.upper(), sheet["annotator_2"].str.strip().str.upper()
    nt = arena != "C"
    res = {
        "n": len(sheet),
        "agree_a1_arena": float((a1 == arena).mean()), "agree_a2_arena": float((a2 == arena).mean()),
        "agree_a1_a2": float((a1 == a2).mean()), "kappa_a1_a2": float(cohen_kappa_score(a1, a2)),
        "agree_a1_arena_nontie": float((a1[nt] == arena[nt]).mean()),
    }
    (d / "agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def errors(run, n):
    test = load_split("test").set_index("id")
    P = pd.read_parquet(OUT_DIR / run / "preds_test.parquet").set_index("id").loc[test.index].to_numpy()
    Ps = pd.read_parquet(OUT_DIR / run / "preds_test_swap.parquet").set_index("id").loc[test.index].to_numpy()
    y = test["label"].to_numpy()
    flip = P.argmax(1) != unswap(Ps).argmax(1)
    conf_wrong = (P.argmax(1) != y) & (P.max(1) > 0.7)
    rng = np.random.default_rng(0)
    pick = lambda m: rng.choice(np.where(m)[0], min(n, m.sum()), replace=False)
    rows = []
    for kind, idx in [("swap_inconsistent", pick(flip)), ("confident_wrong", pick(conf_wrong))]:
        for i in idx:
            r = test.iloc[i]
            rows.append({"kind": kind, "id": test.index[i], "human": LETTER[y[i]], "pred": LETTER[P[i].argmax()],
                         "p_a": P[i, 0], "p_b": P[i, 1], "p_tie": P[i, 2], "len_a": r.len_a, "len_b": r.len_b,
                         "prompt": r.prompt_text[:2000], "resp_a": r.resp_a[:3000], "resp_b": r.resp_b[:3000],
                         "category": ""})
    out = OUT_DIR / run / "error_samples.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"flip rate {flip.mean():.3f}, confident-wrong rate {conf_wrong.mean():.3f}; wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score", "errors"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--run")
    a = ap.parse_args()
    {"sample": lambda: sample(a.n), "score": score, "errors": lambda: errors(a.run, a.n)}[a.cmd]()
