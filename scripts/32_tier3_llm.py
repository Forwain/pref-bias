"""Tier 3b: Qwen2.5-Instruct fine-tuned with LoRA as a 3-way preference classifier (a reward model).

Same family as the Tier 2 zero-shot judge, so "trained on the human votes" vs "zero-shot" is
compared at a fixed model:
  3B  (zero-shot: strongest position bias)  -> position evidence, with and without swap augmentation
  7B  (zero-shot: steepest length slope)    -> length evidence

Matched updates: without augmentation train 2 epochs; with augmentation (2x rows) train 1 epoch.

  python scripts/32_tier3_llm.py --model Qwen/Qwen2.5-3B-Instruct --run tier3_qwen3b --epochs 2
  python scripts/32_tier3_llm.py --model Qwen/Qwen2.5-3B-Instruct --run tier3_qwen3b_swapaug --swap-aug --epochs 1
  python scripts/32_tier3_llm.py --model Qwen/Qwen2.5-7B-Instruct --run tier3_qwen7b_swapaug --swap-aug --epochs 1
  python scripts/32_tier3_llm.py ... --limit 400 --max-steps 20 --pred-limit 200   # smoke test

Input (plain text, the last token is pooled by the classification head):
  Prompt:\n p \n\nResponse A:\n a \n\nResponse B:\n b \n\nWhich response is better?
cut to --max-len tokens; the prompt gets at most a fifth, the two responses share the rest.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.special import softmax
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from prefbias.data import load_split, probs_frame, swap_ab
from prefbias.metrics import accuracy, log_loss
from prefbias.paths import EVAL_SETS, MODEL_DIR, run_dir

MAX_CHARS = 30000  # pre-cut very long texts before tokenising (speed only)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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
    mp, ma, mb, q = (tok(x, add_special_tokens=False)["input_ids"] for x in
                     ["Prompt:\n", "\n\nResponse A:\n", "\n\nResponse B:\n", "\n\nWhich response is better?"])
    budget = max_len - len(mp) - len(ma) - len(mb) - len(q)
    ids = []
    for p, a, b in zip(P, A, B):
        lp = min(len(p), max(128, budget // 5))
        la, lb = allocate(len(a), len(b), budget - lp)
        ids.append(mp + p[:lp] + ma + a[:la] + mb + b[:lb] + q)
    out = {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
    if "label" in df:
        out["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(out)


def compute_metrics(eval_pred):
    logits, y = eval_pred
    P = softmax(logits, axis=1)
    return {"log_loss": log_loss(y, P), "acc": accuracy(y, P)}


@torch.no_grad()
def predict(model, ds, bs, pad_id):
    order = np.argsort([len(x) for x in ds["input_ids"]])[::-1]  # longest first: fail fast on OOM
    out = np.zeros((len(ds), 3), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        seqs = [ds[int(j)]["input_ids"] for j in idx]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def load_base(name, tok):
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, torch_dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--swap-aug", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=2000, help="val rows used for checkpoint selection")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N training rows")
    ap.add_argument("--pred-limit", type=int, default=0, help="smoke test: predict only N rows per set")
    ap.add_argument("--pred-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    train, val = load_split("train"), load_split("val")
    if args.limit:
        train = train.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    if args.swap_aug:
        train = pd.concat([train, swap_ab(train)], ignore_index=True)
    val_sel = val.sample(min(args.eval_n, len(val)), random_state=args.seed)
    print(f"train rows {len(train)} (swap_aug={args.swap_aug}); val for selection {len(val_sel)}", flush=True)

    t0 = time.time()
    ds_tr, ds_va = encode_frame(train, tok, args.max_len), encode_frame(val_sel, tok, args.max_len)
    print(f"tokenised in {time.time() - t0:.0f}s", flush=True)

    model = load_base(args.model, tok)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=2 * args.lora_r,
                      lora_dropout=0.05, target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ckpt = MODEL_DIR / args.run
    targs = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.03, weight_decay=0.01,
        lr_scheduler_type="cosine", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="log_loss",
        greater_is_better=False, logging_steps=25, report_to="none", seed=args.seed,
        dataloader_num_workers=4, group_by_length=True, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    trainer.model.save_pretrained(str(ckpt / "best"))
    tok.save_pretrained(str(ckpt / "best"))

    # Predict from a fresh load (base + saved adapter), not from the in-trainer state.
    del trainer, model
    torch.cuda.empty_cache()
    best = PeftModel.from_pretrained(load_base(args.model, tok), str(ckpt / "best")).cuda().eval()
    out = run_dir(args.run + (f"_n{args.pred_limit}" if args.pred_limit else ""))
    results = {}
    for s in EVAL_SETS:
        df = load_split(s)
        if args.pred_limit:
            df = df.head(args.pred_limit)
        t0 = time.time()
        P = softmax(predict(best, encode_frame(df, tok, args.max_len), args.pred_bs, tok.pad_token_id), axis=1)
        probs_frame(df["id"], P).to_parquet(out / f"preds_{s}.parquet", index=False)
        y = df["label"].to_numpy()
        results[s] = {"log_loss": round(log_loss(y, P), 4), "acc": round(accuracy(y, P), 4),
                      "mean_P": P.mean(0).round(3).tolist(), "seconds": round(time.time() - t0)}
        print(s, results[s], flush=True)
    (out / "params.json").write_text(json.dumps({**vars(args), "train_rows": len(train), "results": results}, indent=2))


if __name__ == "__main__":
    main()
