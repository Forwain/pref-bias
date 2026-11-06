# Bias in LLM preference-prediction models

CSCI 566 (Fall 2026) course project. Data: Kaggle "LLM Classification Finetuning"
(Chatbot Arena human votes). Question: as preference predictors get more capable, does their
reliance on position, length and self-preference shortcuts shrink, or do they exploit them better?

## Layout

| Path | What |
|---|---|
| `src/prefbias/data.py` | parse `train.csv`, prompt-grouped 80/10/10 split, A/B swap |
| `src/prefbias/probes.py` | verbosity probes (`restate`, `filler`) |
| `src/prefbias/features.py` | Tier 1 hand-crafted features |
| `src/prefbias/metrics.py` | log loss, accuracy, ECE, position, length, verbosity, self-preference metrics |
| `scripts/0x_*` | download, prepare, EDA, fetch model weights |
| `scripts/10_tier0.py` | Tier 0 rules (class prior; length lookup table) |
| `scripts/11_tier1.py` | Tier 1 LightGBM + attribution by feature family |
| `scripts/20_tier2_judge.py` | Tier 2 zero-shot Qwen2.5-7B-Instruct judge (vLLM), raw + calibrated, rationales |
| `scripts/30_tier3_deberta.py` | Tier 3 DeBERTa-v3-large (plain / swap-augmented / 20% data) |
| `scripts/40_evaluate.py` | all metrics for all runs -> `outputs/results/results.{csv,md}` |
| `scripts/50_human_eval.py` | annotation sheet, inter-annotator agreement, error-analysis samples |
| `slurm/` | CARC job scripts (account `yzhao010_1531`) |

On CARC: code in `~/pref-bias`, data/models/outputs in `/scratch1/twang607/pref-bias`,
conda env `/scratch1/twang607/envs/pref-bias`.

## Run order (on CARC)

```bash
bash scripts/00_download.sh                 # login node: needs ~/.kaggle token + joined competition
bash scripts/03_fetch_models.sh             # login node: model weights (no HF login needed)
sbatch slurm/cpu_stage.slurm                # prepare, EDA, Tier 0, Tier 1 (no embeddings), evaluate
sbatch slurm/tier1.slurm                    # Tier 1 with embedding features
sbatch slurm/tier2.slurm --limit 300        # judge smoke test
sbatch slurm/tier2.slurm                    # full judge + 200 rationales
sbatch slurm/tier3.slurm --run tier3_deberta
sbatch slurm/tier3.slurm --run tier3_deberta_debias --swap-aug
sbatch slurm/tier3.slurm --run tier3_deberta_debias_frac0.2 --swap-aug --train-frac 0.2
python scripts/40_evaluate.py               # inside the env (source slurm/env.sh)
```

## Evaluation sets

All built from our own held-out `test` split (10% of `train.csv`, grouped by prompt), because the
Kaggle test labels are hidden.

| Set | Change | Used for |
|---|---|---|
| `test` | none | log loss, accuracy, ECE, length slope, self-preference |
| `test_swap` | A and B swapped | position consistency, position bias |
| `test_verbose_restate` | one side gets a recap of its own first sentences | verbosity attack |
| `test_verbose_filler` | one side gets a generic closing paragraph | verbosity attack |

## Metric definitions

- **Position consistency**: share of pairs whose predicted winner is the same after swapping.
- **Position bias**: mean over both orders of P(slot A) - P(slot B); 0 = invariant. Human analogue: P(A wins) - P(B wins).
- **Length slope**: logistic slope of P(A)/(P(A)+P(B)) on log(len A / len B), non-tie pairs. The
  `human_labels` row gives the slope of the human votes; **excess = model - human** is the shortcut measure.
- **Verbosity attack**: mean gain in P(padded side), share of pairs where it rises, share where the predicted winner flips to the padded side.
- **Self-preference** (`--family qwen`): P(Qwen side wins | not tie) minus the human rate.
- `+swapTTA` rows: average of original and un-swapped predictions, an inference-time debias baseline.

## Known caveats

- DeBERTa sees at most `--max-len` tokens (default 512); length beyond the budget is invisible to it.
- The judge reads at most 2,500 tokens per response and 1,000 per prompt.
- The verbosity probes are rule-based; an LLM-paraphrased version would be closer to Zheng et al.
