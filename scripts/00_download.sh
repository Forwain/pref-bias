#!/bin/bash
# Download the competition data (run on the CARC login node: compute nodes may lack internet).
# Needs ~/.kaggle/kaggle.json and having joined the competition on kaggle.com.
set -e
PB_ROOT=${PB_ROOT:-/scratch1/twang607/pref-bias}
mkdir -p "$PB_ROOT/data/raw"
cd "$PB_ROOT/data/raw"
kaggle competitions download -c llm-classification-finetuning
unzip -o llm-classification-finetuning.zip
ls -la
