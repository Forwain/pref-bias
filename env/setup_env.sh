#!/bin/bash
# Build the project env on CARC. Submit to a compute node (login nodes limit threads):
#   sbatch --account=yzhao010_1531 --partition=main --cpus-per-task=8 --mem=32G --time=02:00:00 env/setup_env.sh
set -e
export PATH=/apps/conda/miniforge3/25.11.0-1/condabin:$PATH
eval "$(conda shell.bash hook)"
export PIP_CACHE_DIR=/scratch1/twang607/.pip_cache TMPDIR=/scratch1/twang607/tmp
mkdir -p $TMPDIR
ENV=/scratch1/twang607/envs/pref-bias
mamba create -y -p $ENV python=3.11
conda activate $ENV
pip install --upgrade pip
pip install "vllm==0.6.3.post1"            # pins torch 2.4.0 (cu121) and numpy < 2
pip install "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" "datasets==3.1.0" \
  "sentence-transformers==3.2.1" sentencepiece protobuf tiktoken huggingface_hub \
  lightgbm scikit-learn scipy statsmodels textstat matplotlib seaborn tqdm kaggle six \
  pyyaml pytest tabulate
# newest pyarrow / pandas need numpy >= 2, which vLLM does not allow
pip install "pyarrow==17.0.0" "pandas==2.2.3" "numpy==1.26.4"
python -c "import torch, transformers, vllm, lightgbm, pyarrow, pandas; print('torch', torch.__version__, 'vllm', vllm.__version__, 'pyarrow', pyarrow.__version__)"
echo ENV_DONE
