# Sourced by every job script.
# (`module` is missing when jobs are submitted over a non-login ssh, so use conda's path directly)
export PATH=/apps/conda/miniforge3/25.11.0-1/condabin:$PATH
eval "$(conda shell.bash hook)"
conda activate /scratch1/twang607/envs/pref-bias
export PB_ROOT=/scratch1/twang607/pref-bias
export HF_HOME=$PB_ROOT/hf_cache
export PYTHONPATH=$HOME/pref-bias/src
export TOKENIZERS_PARALLELISM=false
# match BLAS/OpenMP threads to the cores Slurm gave us (default 1 outside a job)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
cd $HOME/pref-bias
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
