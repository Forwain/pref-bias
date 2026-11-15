"""Natural position and length preference in the human labels, per split, with bootstrap CIs.

  python scripts/03_human_bias.py            -> outputs/eda/human_bias.{csv,md}

Position:  P(A wins) - P(B wins)   (0 = no preference for the first-shown answer)
Length:    P(longer wins | not tie), logistic slope of P(A wins) on log(len_A/len_B) (non-tie),
           and tie rate by length-gap quintile.
"""
import numpy as np
import pandas as pd

from prefbias.data import load_split
from prefbias.metrics import bootstrap_ci, human_position_bias, soft_logistic_slope
from prefbias.paths import run_dir


def stats_for(df, B):
    y = df["label"].to_numpy()
    x = df["log_len_ratio"].to_numpy()
    n = len(df)
    nt = y != 2
    diff = df["len_a"].to_numpy() != df["len_b"].to_numpy()

    def longer_wins(idx):
        m = nt[idx] & diff[idx]
        yy, xx = y[idx][m], x[idx][m]
        return float(np.mean(np.where(xx > 0, yy == 0, yy == 1)))

    def slope(idx):
        m = nt[idx]
        return soft_logistic_slope(x[idx][m], (y[idx][m] == 0).astype(float))

    pos = lambda idx: human_position_bias(y[idx])
    row = {"n": n,
           "P(A)": np.mean(y == 0), "P(B)": np.mean(y == 1), "P(tie)": np.mean(y == 2)}
    for name, fn in [("pos_bias", pos), ("P(longer wins|not tie)", longer_wins), ("len_slope", slope)]:
        v = fn(np.arange(n))
        lo, hi = bootstrap_ci(fn, n, B=B)
        row[name] = v
        row[name + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
    # tie rate by |length gap| quintile
    q = pd.qcut(np.abs(x), 5, labels=False, duplicates="drop")
    for k in range(int(q.max()) + 1):
        row[f"tie_q{k + 1}"] = float(np.mean(y[q == k] == 2))
    return row


def main(B=500):
    rows = {}
    for s in ["train", "val", "test"]:
        rows[s] = stats_for(load_split(s), B)
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[s].items()}, flush=True)
    tab = pd.DataFrame(rows).T
    out = run_dir("eda")
    tab.to_csv(out / "human_bias.csv")
    (out / "human_bias.md").write_text(tab.to_markdown(floatfmt=".4f"))
    print(tab.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()
