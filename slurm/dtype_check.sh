#!/bin/bash
# Precision check for the size sweep: re-score the first 300 rows of each split in fp32 (0.5B-3B,
# originally fp16 on V100) and in fp16 (7B, originally bf16). Compare with the full runs.
#   sbatch --job-name=pb-dtype-check --account=yzhao010_1531 --partition=gpu --gres=gpu:1 --mem=48G --time=02:00:00 slurm/dtype_check.sh
set -e
source slurm/env.sh
for s in 0.5B 1.5B 3B; do
  python scripts/20_tier2_judge.py --model Qwen/Qwen2.5-$s-Instruct --tag qwen25_$(echo $s | tr "A-Z." "a-z_")_fp32 --limit 300 --dtype float32
done
python scripts/20_tier2_judge.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b_fp16 --limit 300 --dtype float16
