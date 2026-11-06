#!/bin/bash
#SBATCH --job-name=pb-tier1
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Tier 1 with sentence-embedding similarity features (GPU only for the embeddings).
set -e
source slurm/env.sh
python scripts/11_tier1.py --run tier1_lgbm
