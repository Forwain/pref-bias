#!/bin/bash
# Pre-download model weights on the login node into the shared HF cache (all ungated, no login needed).
set -e
PB_ROOT=${PB_ROOT:-/scratch1/twang607/pref-bias}
export HF_HOME=$PB_ROOT/hf_cache
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download microsoft/deberta-v3-large
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
