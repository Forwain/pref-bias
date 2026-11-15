# Literature notes: bias in preference prediction / LLM-as-judge

Collected 2026-11-10. Numbers marked (verified) were read from the paper; others come from
abstracts or summaries and must be checked against the paper before they go in the report.

## 1. Position bias

**Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 D&B.** arXiv:2306.05685
- Defines position, verbosity and self-enhancement bias for LLM judges. This is the paper the proposal extends.
- Table 2 (verified), swap A/B, default prompt:

  | judge | consistent | favors first | favors second |
  |---|---|---|---|
  | Claude-v1 | 23.8% | 75.0% | 0.0% |
  | GPT-3.5 | 46.2% | 50.0% | 1.2% |
  | GPT-4 | 65.0% | 30.0% | 5.0% |

- Mitigations (verified): swap and call a tie when the two verdicts disagree; few-shot judge prompt raised GPT-4 consistency from 65.0% to 77.5%; CoT / reference-guided prompts cut math errors from 70% to 30% / 15%.
- Agreement (verified, Table 5): GPT-4 vs human 66% with ties (85% without); human vs human 63% (81%). This is the ceiling to compare our ~47% three-way accuracy against.

**Wang et al. 2023, "Large Language Models are not Fair Evaluators".** arXiv:2305.17926, ACL 2024
- Ranking can be flipped just by reordering. Balanced Position Calibration (BPC) scores both orders and averages them, which is the same as our +swapTTA. Multiple Evidence Calibration (explain first, then score) is also proposed.
- GPT-4 favors the first answer; ChatGPT favors the second, so the direction depends on the judge.

**Shi et al. 2024, "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge".** arXiv:2406.07791
- 9 judges, 22 tasks, about 80K evaluations. Three metrics: repetition stability, position consistency, preference fairness (direction).
- Consistency and fairness are different things: a judge can be consistent but unfair, or fair but inconsistent. This matches our split between pos_consistency and pos_bias.
- Position consistency rises with the quality gap between the answers; close pairs are the hardest. We should reproduce this with a difficulty-binned plot.

## 2. Length, verbosity and style

**LMSYS 2024, "Does style matter? Disentangling style and substance in Chatbot Arena" (blog, 2024-08-28).** lmsys.org/blog/2024-08-28-style-control
- Bradley-Terry with model identity plus style covariates: length difference normalized as (lenA-lenB)/(lenA+lenB), and counts of headers, bold and lists.
- Length is the dominant style factor (coefficient about 0.25); markdown effects are second-order. Rankings change after control.
- Use the same covariates for our human-label analysis (RQ10) so the numbers are comparable.

**Dubois et al. 2024, "Length-Controlled AlpacaEval".** arXiv:2404.04475, COLM 2024
- Fits a GLM of judge preference on length difference and other terms, then predicts the preference at zero length difference. Spearman with Chatbot Arena rises from 0.94 to 0.98, and the metric becomes harder to game by verbosity.
- Could serve as a "length-controlled" score for each of our tiers.

**Singhal et al. 2023, "A Long Way to Go: Investigating Length Correlations in RLHF".** arXiv:2310.03716, COLM 2024
- Reward models trained on open preference data correlate strongly with length. A length-only reward reproduces most RLHF gains.
- Parallel to our finding that Tier 0 (length only) captures a large part of Tier 1's gain.

**Saito et al. 2023, "Verbosity Bias in Preference Labeling by Large Language Models".** arXiv:2310.10076
- GPT-4 prefers longer answers more than humans do. Proposes a verbosity-bias metric based on accuracy parity.
- Matches our "excess over human slope" idea.

**Chen et al. 2024 (ODIN), "Disentangled Reward Mitigates Hacking in RLHF".** arXiv:2402.07319, ICML 2024
- Two reward heads: one is trained to carry length, the other is decorrelated from length. Only the content head is used.
- A possible extra mitigation condition for Tier 3.

## 3. Human-label bias

**Chen et al. 2024, "Humans or LLMs as the Judge? A Study on Judgement Bias".** arXiv:2402.10669, EMNLP 2024
- Perturbation framework that needs no ground truth: add fake references, rich formatting, or factual errors, then measure Attack Success Rate. Both humans and LLMs are vulnerable; humans show verbosity, beauty (format) and fallacy-oversight bias.
- This is the methodological precedent for our verbosity probes. A "format padding" probe would match their beauty bias.

**Hosking et al. 2024, "Human Feedback is not Gold Standard".** arXiv:2309.16349, ICLR 2024
- Overall preference scores under-weight factuality. Assertiveness skews perceived errors. Training on human preference raises assertiveness.
- Supports treating human labels as a reference, not ground truth (see the report discussion).

## 4. Self-preference

**Panickssery et al. 2024, "LLM Evaluators Recognize and Favor Their Own Generations".** arXiv:2404.13076, NeurIPS 2024
- GPT-4 and Llama 2 favor their own summaries. Self-recognition ability correlates with self-preference strength.
- For us: Qwen2.5 judging older Qwen models in Arena is "family" preference, a weaker setting. Expect a small effect and report it with CIs.

## 5. Broader bias benchmarks

**Ye et al. 2025, "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (CALM).** arXiv:2410.02736, ICLR 2025
- 12 bias types, automated input perturbations, robustness rate (share of verdicts unchanged).

**Park et al. 2024, "OffsetBias: Leveraging Debiased Data for Tuning Evaluators".** arXiv:2407.06551, Findings of EMNLP 2024
- 6 judge bias types (length, concreteness, empty reference, content continuation, nested instruction, ...). Fine-tuning on debiased pairs is used as the mitigation.

## 6. Kaggle LMSYS competition (same data)

- Top solutions fine-tuned Gemma-2-9B with LoRA, distilled from larger models, and averaged predictions over both A/B orders at test time (Nebius review; Kaggle write-ups).
- The 4th-place-derived `pairjudge` package reports that 29.2% of verdicts flip on swap (2K held-out pairs), and that swap averaging improves log loss while leaving accuracy flat. This is the same pattern as our Tier 1 and Tier 2 +swapTTA rows.
- The Arena team's Kaggle post describes the hidden test set of about 25K battles.

## 7. Engineering note: DeBERTa-v3

- No authoritative source. One forum thread suggests a diagnostic: stable in fp32 but not fp16 means a precision problem; unstable in both means the data, loss or optimizer setup is wrong.
- A loss stuck at ln(K) means the model never learned. The usual fixes are more warmup, a different seed or learning rate, and layer-wise learning rate decay.
