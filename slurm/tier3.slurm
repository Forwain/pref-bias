#!/bin/bash
#SBATCH --job-name=pb-tier3
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# DeBERTa-v3-large fine-tuning. Pass the script arguments after the job file:
#   sbatch slurm/tier3.slurm --run tier3_deberta
#   sbatch slurm/tier3.slurm --run tier3_deberta_debias --swap-aug
#   sbatch slurm/tier3.slurm --run tier3_deberta_debias_frac0.2 --swap-aug --train-frac 0.2
# Any A40 / A100 / L40S (>= 40 GB) is enough at max-len 512.
set -e
source slurm/env.sh
python scripts/30_tier3_deberta.py "$@"
