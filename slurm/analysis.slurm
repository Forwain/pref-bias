#!/bin/bash
#SBATCH --job-name=pb-analysis
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/twang607/pref-bias/logs/%x-%j.out
# Results table + surface-cue explainability over every finished run.
#   sbatch slurm/analysis.slurm
#   sbatch --dependency=afterany:<jobs> slurm/analysis.slurm
set -e
source slurm/env.sh
python scripts/40_evaluate.py
python scripts/41_surface_explain.py
