"""
viz/plot_results.py

Plots the training curves from the progress.csv files that
Stable-Baselines3 writes during training.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from viz.plotting_utils import set_style

# The runs that belong in the figure, and how they are labelled.
RUN_LABELS = {
    "PPO_primary_seed42": "PPO (primary)",
    "PPO_conditioned_seed42": "PPO (conditioned, 1M steps)",
    "PPO_conditioned_150k_seed42": "PPO (conditioned, 150k steps)",
}
RUN_COLORS = {
    "PPO_primary_seed42": "#1f77b4",
    "PPO_conditioned_seed42": "#0d9488",
    "PPO_conditioned_150k_seed42": "#94a3b8",
}

STEP_COLUMN = "time/total_timesteps"
REWARD_COLUMN = "rollout/ep_rew_mean"


def plot_training_curves():
    curves_dir = PROJECT_ROOT / "results" / "curves"
    plot_path = PROJECT_ROOT / "results" / "plots" / "training_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    set_style()
    plt.figure(figsize=(10, 6))

    plotted = 0
    for run_dir in sorted(curves_dir.iterdir() if curves_dir.exists() else []):
        if not run_dir.is_dir():
            continue
        curve_path = run_dir / "progress.csv"
        if not curve_path.exists():
            continue
        name = run_dir.name
        if not name.startswith("PPO"):
            continue

        df = pd.read_csv(curve_path)
        if df.empty or REWARD_COLUMN not in df.columns or STEP_COLUMN not in df.columns:
            continue

        plt.plot(df[STEP_COLUMN], df[REWARD_COLUMN],
                 label=RUN_LABELS.get(name, name),
                 color=RUN_COLORS.get(name, "#64748b"),
                 alpha=0.9, linewidth=2)
        plotted += 1

    plt.title("Training Curves: Mean Episode Reward")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Normalized Episode Reward Mean")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path} ({plotted} runs)")


if __name__ == "__main__":
    plot_training_curves()
