#!/bin/bash
#SBATCH --job-name=pb-t3llm
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Qwen2.5 + LoRA reward model (bf16 needs an Ampere+ GPU). Script arguments follow the job file:
#   sbatch slurm/tier3_llm.slurm --model Qwen/Qwen2.5-3B-Instruct --run tier3_qwen3b --epochs 2
set -e
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/32_tier3_llm.py "$@"
