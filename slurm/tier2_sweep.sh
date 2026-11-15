#!/bin/bash
# Submit the Tier 2 size sweep: Qwen2.5-{0.5B,1.5B,3B,14B}-Instruct, same prompt as the 7B run.
# A CPU job downloads the weights first; each judge job waits for it.
# Each judge job: full scoring on all eval sets + 200 rationales (same sample as the 7B run).
#   bash slurm/tier2_sweep.sh
cd "$(dirname "$0")/.."
dl=$(sbatch --parsable --job-name=pb-dl --account=yzhao010_1531 --partition=main --cpus-per-task=4 \
  --mem=16G --time=04:00:00 --output=/scratch1/twang607/pref-bias/logs/%x-%j.out \
  --wrap 'source slurm/env.sh; for s in 0.5B 1.5B 3B 14B; do huggingface-cli download Qwen/Qwen2.5-$s-Instruct; done')
echo "download job $dl"
for s in 0.5B 1.5B 3B 14B; do
  tag=qwen25_$(echo $s | tr 'A-Z.' 'a-z_')
  sbatch --dependency=afterok:$dl --job-name=pb-t2-$s slurm/tier2.slurm \
    --model Qwen/Qwen2.5-$s-Instruct --tag $tag --rationale-n 200
done
