"""
config/paths.py

Where the project keeps things.

The model and result directories are referred to from the training scripts,
the evaluation scripts, the plots and the dashboard. Keeping the names in one
place means a renamed artifact is a one line change.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CURVES_DIR = RESULTS_DIR / "curves"
WORKERS_DIR = RESULTS_DIR / "workers"
PAPERS_DIR = PROJECT_ROOT / "docs" / "papers"

# The policies this project ships with.
PRIMARY_MODEL_DIR = MODELS_DIR / "PPO_seed42"
CONDITIONED_MODEL_DIR = MODELS_DIR / "PPO_conditioned_seed42"
