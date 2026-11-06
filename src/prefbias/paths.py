"""Where data, models and outputs live. Override the root with PB_ROOT."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PB_ROOT", "/scratch1/twang607/pref-bias"))
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

# Evaluation sets built by scripts/01_prepare.py.
# "test" is our own held-out split of the Kaggle train.csv (the Kaggle test set has no labels).
EVAL_SETS = ["val", "test", "test_swap", "test_verbose_restate", "test_verbose_filler"]


def run_dir(run_name: str) -> Path:
    d = OUT_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d
