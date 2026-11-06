#!/bin/bash
#SBATCH --job-name=pb-cpu
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Prepare data, EDA, Tier 0, Tier 1 without embeddings, then evaluate.
#   sbatch slurm/cpu_stage.slurm
set -e
source slurm/env.sh
python scripts/01_prepare.py
python scripts/02_eda.py
python scripts/10_tier0.py
python scripts/11_tier1.py --no-emb --run tier1_lgbm_noemb
python scripts/40_evaluate.py
