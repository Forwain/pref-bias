#!/bin/bash
#SBATCH --job-name=pb-tier2
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Zero-shot Qwen2.5-7B-Instruct judge.
#   sbatch slurm/tier2.slurm                    # full run + 200 rationales
#   sbatch slurm/tier2.slurm --limit 300        # quick check (extra args go to the script)
set -e
source slurm/env.sh
ARGS=${@:---rationale-n 200}
python scripts/20_tier2_judge.py $ARGS
