"""Which surface properties do human voters reward, beyond length? (train split, labels only)

  python scripts/04_human_style_prefs.py      -> outputs/eda/human_style_prefs.md

For each property, on non-tie pairs:
  raw win rate   P(side that has it wins | exactly one side has it)
  same-length    the same, only on pairs with |log(len_A/len_B)| < 0.2
  logit coef     y=A wins ~ x            (x = has_A - has_B, or z-scored A-B difference)
  coef | length  y ~ x + log length ratio
  coef | all     one joint model with every property + length
Also: tie rate vs. how similar the two answers are, and when both refuse.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from prefbias.data import load_split
from prefbias.paths import PROC_DIR, run_dir

BINARY = {"bullets": "n_bullets", "headers": "n_headers", "bold": "n_bold", "code block": "n_code",
          "url": "n_urls", "asks question": "n_question", "exclamation": "n_exclaim",
          "refusal": "refusal", "empty": "empty"}
CONT = {"more paragraphs (log)": "n_paragraphs", "easier to read (Flesch)": "flesch",
        "lexical diversity": "uniq_ratio", "longer words": "avg_word_len",
        "relevance to prompt (emb sim)": "sim_prompt"}


def logit(y, X):
    r = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    return r.params, r.bse, r.pvalues


def main():
    df = load_split("train")
    F = pd.read_parquet(PROC_DIR / "feats_train_emb.parquet")
    y = df["label"].to_numpy()
    nt = y != 2
    yA = (y[nt] == 0).astype(float)
    L = F["d_log_chars"].to_numpy()
    near = np.abs(L) < 0.2

    xs = {}
    rows = []
    for name, c in BINARY.items():
        ha, hb = (F[f"a_{c}"] > 0).to_numpy(), (F[f"b_{c}"] > 0).to_numpy()
        x = ha.astype(float) - hb.astype(float)
        xs[name] = x
        one = nt & (x != 0)
        win = np.where(x[one] > 0, y[one] == 0, y[one] == 1)
        one_n = one & near
        win_n = np.where(x[one_n] > 0, y[one_n] == 0, y[one_n] == 1)
        rows.append({"property": name, "type": "binary",
                     "share of pairs (one side only)": (x != 0).mean(),
                     "raw win rate": win.mean(), "n": int(one.sum()),
                     "same-length win rate": win_n.mean() if one_n.sum() else np.nan,
                     "n same-length": int(one_n.sum())})
    for name, c in CONT.items():
        a, b = F[f"a_{c}"].to_numpy(float), F[f"b_{c}"].to_numpy(float)
        d = np.log1p(a) - np.log1p(b) if c == "n_paragraphs" else a - b
        xs[name] = d / d[nt].std()
        rows.append({"property": name, "type": "z-scored diff"})
    xs["log length ratio"] = L / L[nt].std()
    rows.append({"property": "log length ratio", "type": "z-scored diff"})

    allX = pd.DataFrame({k: v[nt] for k, v in xs.items()})
    pj, sj, pvj = logit(yA, allX.to_numpy())
    for i, r in enumerate(rows):
        k = r["property"]
        p, s, _ = logit(yA, allX[[k]].to_numpy())
        r["coef"] = p[1]
        if k != "log length ratio":
            p2, s2, _ = logit(yA, allX[[k, "log length ratio"]].to_numpy())
            r["coef | length"], r["se | length"] = p2[1], s2[1]
        j = list(allX.columns).index(k) + 1
        r["coef | all"], r["se | all"], r["p | all"] = pj[j], sj[j], pvj[j]
    tab = pd.DataFrame(rows)

    # ties
    sim = F["sim_ab"].to_numpy()
    qs = pd.qcut(sim, 5, labels=False)
    tie_sim = {f"sim_ab q{k + 1}": float(np.mean(y[qs == k] == 2)) for k in range(5)}
    both_ref = (F["a_refusal"] > 0).to_numpy() & (F["b_refusal"] > 0).to_numpy()
    one_ref = (F["a_refusal"] > 0).to_numpy() ^ (F["b_refusal"] > 0).to_numpy()
    tie_ref = {"tie | both refuse": float(np.mean(y[both_ref] == 2)), "n both": int(both_ref.sum()),
               "tie | one refuses": float(np.mean(y[one_ref] == 2)),
               "tie | overall": float(np.mean(y == 2))}
    mt = df["n_turns"].to_numpy() > 1
    tie_turn = {"tie | single-turn": float(np.mean(y[~mt] == 2)), "tie | multi-turn": float(np.mean(y[mt] == 2))}

    out = run_dir("eda")
    md = ["## Human preference for surface properties (train, non-tie)\n",
          tab.to_markdown(index=False, floatfmt=".3f"),
          "\n\n## Tie rate\n", pd.Series({**tie_sim, **tie_ref, **tie_turn}).to_markdown(floatfmt=".3f")]
    (out / "human_style_prefs.md").write_text("\n".join(md))
    tab.to_csv(out / "human_style_prefs.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
