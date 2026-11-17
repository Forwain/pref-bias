#!/bin/bash
# Smoke test for scripts/32_tier3_llm.py: 20 steps on 400 rows, predict 100 rows per set.
set -e
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in 7B 3B; do
  python scripts/32_tier3_llm.py --model Qwen/Qwen2.5-$m-Instruct --run smoke_qwen$m --swap-aug \
    --limit 400 --max-steps 20 --eval-steps 10 --eval-n 100 --pred-limit 100
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
