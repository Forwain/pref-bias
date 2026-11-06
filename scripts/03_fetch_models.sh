#!/bin/bash
# Pre-download model weights on the login node (internet) into the shared HF cache.
# Llama 3 is gated: accept the license on huggingface.co and run `huggingface-cli login` first.
set -e
PB_ROOT=${PB_ROOT:-/scratch1/twang607/pref-bias}
export HF_HOME=$PB_ROOT/hf_cache
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download microsoft/deberta-v3-large
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"
