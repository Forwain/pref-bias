#!/bin/bash
#SBATCH --job-name=pb-t3pred
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Re-predict every eval set from saved best checkpoints (fixes the bad in-trainer predictions).
#   sbatch slurm/tier3_predict.slurm tier3_deberta_debias tier3_deberta_debias_frac0.2
set -e
source slurm/env.sh
for r in "$@"; do
  python scripts/31_tier3_predict.py --ckpt $PB_ROOT/models/$r/best --run $r
done
