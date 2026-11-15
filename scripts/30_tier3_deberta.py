"""Tier 3: DeBERTa-v3-large fine-tuned as a 3-way classifier (a small reward model).

Conditions:
  tier3_deberta                    plain fine-tuning
  tier3_deberta_debias             + swap augmentation (every pair also shown with A/B swapped)
  tier3_deberta_debias_frac0.2     swap augmentation, 20% of training pairs (data-scarcity check)

  python scripts/30_tier3_deberta.py --run tier3_deberta
  python scripts/30_tier3_deberta.py --run tier3_deberta_debias --swap-aug
  python scripts/30_tier3_deberta.py --run tier3_deberta_debias_frac0.2 --swap-aug --train-frac 0.2

Input: [CLS] Prompt: p [SEP] Response A: a [SEP] Response B: b [SEP], cut to --max-len tokens.
The token budget is shared so a short response leaves room for a long one. Truncation hides
length beyond the budget, which itself matters for the length-bias reading (see README).
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from prefbias.data import load_split, probs_frame, swap_ab
from prefbias.metrics import accuracy, log_loss
from prefbias.paths import EVAL_SETS, MODEL_DIR, run_dir

MAX_CHARS = 20000  # pre-cut very long texts before tokenising (speed only)


def allocate(n_a, n_b, budget):
    half = budget // 2
    if n_a <= half:
        return n_a, min(n_b, budget - n_a)
    if n_b <= half:
        return min(n_a, budget - n_b), n_b
    return half, budget - half


def encode_frame(df, tok, max_len):
    enc = lambda s: tok([t[:MAX_CHARS] for t in s], add_special_tokens=False)["input_ids"]
    P, A, B = enc(df["prompt_text"].tolist()), enc(df["resp_a"].tolist()), enc(df["resp_b"].tolist())
    mp, ma, mb = (tok(x, add_special_tokens=False)["input_ids"] for x in ["Prompt:", "Response A:", "Response B:"])
    cls, sep = tok.cls_token_id, tok.sep_token_id
    fixed = 4 + len(mp) + len(ma) + len(mb)
    budget = max_len - fixed
    ids = []
    for p, a, b in zip(P, A, B):
        lp = min(len(p), max(64, budget // 5))
        la, lb = allocate(len(a), len(b), budget - lp)
        ids.append([cls] + mp + p[:lp] + [sep] + ma + a[:la] + [sep] + mb + b[:lb] + [sep])
    out = {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
    if "label" in df:
        out["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(out)


def compute_metrics(eval_pred):
    logits, y = eval_pred
    P = softmax(logits, axis=1)
    return {"log_loss": log_loss(y, P), "acc": accuracy(y, P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--swap-aug", action="store_true")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    train, val = load_split("train"), load_split("val")
    if args.train_frac < 1:
        train = train.sample(frac=args.train_frac, random_state=args.seed).reset_index(drop=True)
    if args.swap_aug:
        train = pd.concat([train, swap_ab(train)], ignore_index=True)
    print(f"train rows {len(train)} (swap_aug={args.swap_aug}, frac={args.train_frac})")

    ds_tr, ds_va = encode_frame(train, tok, args.max_len), encode_frame(val, tok, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    out = run_dir(args.run)
    ckpt = MODEL_DIR / args.run
    targs = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 4,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.1, weight_decay=0.01,
        lr_scheduler_type="cosine", bf16=True, gradient_checkpointing=args.grad_ckpt,
        eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="log_loss",
        greater_is_better=False, logging_steps=50, report_to="none", seed=args.seed,
        dataloader_num_workers=4, group_by_length=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(ckpt / "best"))
    tok.save_pretrained(str(ckpt / "best"))

    # Predict from a fresh load of the saved best model. trainer.predict() after training gave
    # predictions far worse than the logged best eval (1.19 vs 1.013 on val) while a fresh load
    # reproduces 1.013, so the in-trainer model state is not trusted here.
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    spec = spec_from_file_location("t3pred", Path(__file__).with_name("31_tier3_predict.py"))
    t3pred = module_from_spec(spec); spec.loader.exec_module(t3pred)
    del trainer, model
    torch.cuda.empty_cache()
    best = AutoModelForSequenceClassification.from_pretrained(str(ckpt / "best")).cuda().eval()
    results = {}
    for s in EVAL_SETS:
        df = load_split(s)
        logits = t3pred.predict(best, encode_frame(df, tok, args.max_len), args.bs * 4, "fp32")
        P = softmax(logits, axis=1)
        probs_frame(df["id"], P).to_parquet(out / f"preds_{s}.parquet", index=False)
        if "label" in df:
            results[s] = {"log_loss": log_loss(df["label"].to_numpy(), P), "acc": accuracy(df["label"].to_numpy(), P)}
    (out / "params.json").write_text(json.dumps({**vars(args), "train_rows": len(train), "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
