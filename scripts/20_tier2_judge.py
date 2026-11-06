"""Tier 2: zero-shot LLM-as-judge (Qwen2.5-7B-Instruct via vLLM), no fine-tuning.

Scoring mode: the judge answers with one letter (A / B / C=tie); we read the log-probabilities
of those three tokens at the first generated position and renormalise -> P(A), P(B), P(tie).
Two runs are written:
  tier2_<tag>      raw judge probabilities
  tier2_<tag>_cal  temperature + per-class bias fitted on the val split (4 numbers)
Rationale mode (--rationale-n N): on N random test pairs the judge first explains, then gives
"[[A]]", "[[B]]" or "[[C]]"; saved as CSV for the qualitative analysis.

  python scripts/20_tier2_judge.py --limit 300          # quick check on a small sample
  python scripts/20_tier2_judge.py --rationale-n 200    # full run
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from prefbias.data import load_split, probs_frame
from prefbias.metrics import accuracy, log_loss
from prefbias.paths import EVAL_SETS, run_dir

SYSTEM = (
    "You are an impartial judge. You will see a conversation between a user and two AI assistants, "
    "A and B. Decide which assistant gave the better responses: consider helpfulness, relevance, "
    "accuracy, depth and clarity. Do not let the order in which the responses are shown, their "
    "length, or the assistants' names affect your decision."
)
SCORE_INSTR = "Which assistant is better? Answer with a single letter: A, B, or C for a tie."
RATIONALE_INSTR = (
    "First explain your comparison briefly. Then give your final verdict strictly in this format: "
    "\"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, \"[[C]]\" for a tie."
)


def truncate(tok, text, max_tokens):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens]) + "\n...[truncated]"


def render_pair(row, tok, max_resp_tokens, max_prompt_tokens):
    """Show the conversation turn by turn for both assistants."""
    p, a, b = list(row["turns_prompt"]), list(row["turns_a"]), list(row["turns_b"])
    n = max(len(p), 1)
    per_p, per_r = max_prompt_tokens // n, max_resp_tokens // n
    parts = []
    for name, resp in [("A", a), ("B", b)]:
        lines = [f"<|The Start of Assistant {name}'s Conversation|>"]
        for i in range(len(p)):
            lines.append(f"### User:\n{truncate(tok, p[i] or '', per_p)}")
            r = resp[i] if i < len(resp) else ""
            lines.append(f"### Assistant {name}:\n{truncate(tok, r or '(no response)', per_r)}")
        lines.append(f"<|The End of Assistant {name}'s Conversation|>")
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)


def build_prompts(df, tok, instr, args):
    out = []
    for _, row in df.iterrows():
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": render_pair(row, tok, args.max_resp_tokens, args.max_prompt_tokens)
                 + "\n\n" + instr}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def letter_logprobs(outputs):
    """(n, 3) log-probs for A, B, C taken from the top-k list at the first generated token."""
    L = np.full((len(outputs), 3), -50.0)
    for i, o in enumerate(outputs):
        top = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        for lp in top.values():
            t = (lp.decoded_token or "").strip().strip("[").upper()
            if t in ("A", "B", "C"):
                k = "ABC".index(t)
                L[i, k] = max(L[i, k], lp.logprob)
    return L


def fit_calibration(L, y):
    """P = softmax(L / T + bias); fit T > 0 and bias (3) on val by NLL."""
    def nll(w):
        T, bias = np.exp(w[0]), w[1:]
        lp = log_softmax(L / T + bias, axis=1)
        return -lp[np.arange(len(y)), y].mean()
    res = minimize(nll, np.zeros(4), method="Nelder-Mead", options={"maxiter": 4000})
    return float(np.exp(res.x[0])), res.x[1:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="qwen25_7b")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows of each set")
    ap.add_argument("--max-resp-tokens", type=int, default=2500)
    ap.add_argument("--max-prompt-tokens", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--rationale-n", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    score_sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    tag = args.tag + (f"_n{args.limit}" if args.limit else "")
    d_raw, d_cal = run_dir(f"tier2_{tag}"), run_dir(f"tier2_{tag}_cal")
    raw_logits = {}
    for s in EVAL_SETS:
        df = load_split(s)
        if args.limit:
            df = df.head(args.limit)
        outs = llm.generate(build_prompts(df, tok, SCORE_INSTR, args), score_sp)
        L = letter_logprobs(outs)
        raw_logits[s] = (df, L)
        np.save(d_raw / f"logits_{s}.npy", L)
        probs_frame(df["id"], softmax(L, axis=1)).to_parquet(d_raw / f"preds_{s}.parquet", index=False)
        missing = float(np.mean((L <= -50).all(1)))
        print(f"{s}: {len(df)} rows, no A/B/C token in top-20 for {missing:.2%}")

    df_val, L_val = raw_logits["val"]
    T, bias = fit_calibration(L_val, df_val["label"].to_numpy())
    (d_cal / "params.json").write_text(json.dumps({"T": T, "bias": bias}))
    for s, (df, L) in raw_logits.items():
        P = softmax(L / T + np.array(bias), axis=1)
        probs_frame(df["id"], P).to_parquet(d_cal / f"preds_{s}.parquet", index=False)
    df_t, L_t = raw_logits["test"]
    y = df_t["label"].to_numpy()
    print("test raw: logloss %.4f acc %.4f" % (log_loss(y, softmax(L_t, 1)), accuracy(y, softmax(L_t, 1))))
    Pc = softmax(L_t / T + np.array(bias), 1)
    print("test cal: logloss %.4f acc %.4f (T=%.3f)" % (log_loss(y, Pc), accuracy(y, Pc), T))

    if args.rationale_n:
        test = load_split("test").sample(args.rationale_n, random_state=0).reset_index(drop=True)
        outs = llm.generate(build_prompts(test, tok, RATIONALE_INSTR, args),
                            SamplingParams(max_tokens=768, temperature=0.0))
        texts = [o.outputs[0].text for o in outs]
        verdict = [(re.findall(r"\[\[([ABC])\]\]", t) or ["?"])[-1] for t in texts]
        pd.DataFrame({
            "id": test["id"], "model_a": test["model_a"], "model_b": test["model_b"],
            "human_label": test["label"].map({0: "A", 1: "B", 2: "C"}), "judge_verdict": verdict,
            "len_a": test["len_a"], "len_b": test["len_b"], "rationale": texts,
            "category": "",  # filled in by hand during error analysis
        }).to_csv(d_raw / "rationales.csv", index=False)
        print("rationales saved")


if __name__ == "__main__":
    main()
