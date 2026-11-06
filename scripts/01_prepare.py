"""Parse train.csv, split 80/10/10 by prompt, and build the probe sets.

Writes data/processed/{train,val,test,test_swap,test_verbose_restate,test_verbose_filler}.parquet
"""
import json

from prefbias.data import load_raw, split_of, swap_ab
from prefbias.paths import PROC_DIR
from prefbias.probes import make_verbose_probe


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw("train.csv")
    df["split"] = df["prompt_text"].map(split_of)
    sets = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    sets["test_swap"] = swap_ab(sets["test"])
    sets["test_verbose_restate"] = make_verbose_probe(sets["test"], "restate", seed=1)
    sets["test_verbose_filler"] = make_verbose_probe(sets["test"], "filler", seed=2)

    summary = {}
    for name, d in sets.items():
        d.to_parquet(PROC_DIR / f"{name}.parquet", index=False)
        summary[name] = {"rows": len(d)}
        if "label" in d:
            summary[name]["label_dist"] = d["label"].value_counts(normalize=True).sort_index().round(4).tolist()
    print(json.dumps(summary, indent=2))
    (PROC_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
