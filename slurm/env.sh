# Sourced by every job script.
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /scratch1/twang607/envs/pref-bias
export PB_ROOT=/scratch1/twang607/pref-bias
export HF_HOME=$PB_ROOT/hf_cache
export PYTHONPATH=$HOME/pref-bias/src
export TOKENIZERS_PARALLELISM=false
cd $HOME/pref-bias
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
