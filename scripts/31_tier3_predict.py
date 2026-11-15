"""Predict with a saved Tier 3 checkpoint (no training).

  python scripts/31_tier3_predict.py --ckpt models/tier3_deberta_debias/checkpoint-4500 --run tier3_deberta_debias
  python scripts/31_tier3_predict.py --ckpt ... --sets val --dtype bf16    # quick check only

Writes preds_<set>.parquet into outputs/<run> unless --dry.
"""
import argparse
import json

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prefbias.data import load_split, probs_frame
from prefbias.metrics import accuracy, log_loss
from prefbias.paths import EVAL_SETS, MODEL_DIR, run_dir

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("t3", pathlib.Path(__file__).with_name("30_tier3_deberta.py"))
t3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(t3)


@torch.no_grad()
def predict(model, ds, bs, dtype):
    order = np.argsort([len(x) for x in ds["input_ids"]])  # sort by length for speed, restore after
    out = np.zeros((len(ds), 3), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        seqs = [ds[int(j)]["input_ids"] for j in idx]
        L = max(map(len, seqs))
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == "bf16")):
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--sets", nargs="+", default=EVAL_SETS)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).cuda().eval()
    out = run_dir(args.run) if (args.run and not args.dry) else None
    results = {}
    for s in args.sets:
        df = load_split(s)
        P = softmax(predict(model, t3.encode_frame(df, tok, args.max_len), args.bs, args.dtype), axis=1)
        if out is not None:
            probs_frame(df["id"], P).to_parquet(out / f"preds_{s}.parquet", index=False)
        y = df["label"].to_numpy()
        results[s] = {"log_loss": round(log_loss(y, P), 4), "acc": round(accuracy(y, P), 4),
                      "mean_P": P.mean(0).round(3).tolist()}
        print(s, results[s], flush=True)
    if out is not None:
        (out / "predict_params.json").write_text(json.dumps({**vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
