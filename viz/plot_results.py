"""
viz/plot_results.py

Plots the training curves from the progress.csv files that
Stable-Baselines3 writes during training.

Every run of the project writes its curve to results/run_<run_name>/log:
  PPO_seed42                   the production policy, the weight and the
                               scenario drawn per episode
  ablation_w2_<w2>_seed<seed>  the ablation grid, one policy per energy
                               weight with that weight pinned
  transfer_<arm>_seed42        the transfer experiment
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.config import PLOTS_DIR, RESULTS_DIR, run_dir, run_log
from viz.plotting_utils import set_style

# The runs that are labelled by name, and how they are labelled.
RUN_LABELS = {
    "PPO_seed42": "PPO (production policy)",
}
RUN_COLORS = {
    "PPO_seed42": "#1f77b4",
}

# The ablation grid is a family of runs, not a single run: the individual
# curves are drawn faintly and the mean over the grid carries the label.
ABLATION_PREFIX = "ablation_w2_"
ABLATION_COLOR = "#ea580c"
TRANSFER_COLORS = {"source": "#0d9488", "scratch": "#94a3b8"}

STEP_COLUMN = "time/total_timesteps"
REWARD_COLUMN = "rollout/ep_rew_mean"


def read_curve(log_dir):
    """The reward curve of one run, or None when it has nothing to plot."""
    curve_path = log_dir / "progress.csv"
    if not curve_path.exists():
        return None

    df = pd.read_csv(curve_path)
    if df.empty or REWARD_COLUMN not in df.columns or STEP_COLUMN not in df.columns:
        return None
    return df


def ablation_mean(curves):
    """Mean reward curve over the ablation grid.

    The runs of the grid do not all have the same length, so they are read
    on a common step grid that stops at the shortest of them. Nothing is
    extrapolated past the end of a run.
    """
    shortest = min(df[STEP_COLUMN].max() for _, df in curves)
    grid = np.linspace(0.0, shortest, 200)
    rewards = np.vstack([np.interp(grid, df[STEP_COLUMN], df[REWARD_COLUMN])
                         for _, df in curves])
    return grid, rewards.mean(axis=0)


def transfer_label(name):
    """Label of one arm of the transfer experiment."""
    if "source" in name:
        return "Transfer: source policy"
    if "scratch" in name:
        return "Transfer: trained from scratch"
    return f"Transfer ({name})"


def transfer_color(name):
    """Colour of one arm of the transfer experiment."""
    if "source" in name:
        return TRANSFER_COLORS["source"]
    if "scratch" in name:
        return TRANSFER_COLORS["scratch"]
    return "#64748b"


def plot_training_curves():
    plot_path = PLOTS_DIR / "training_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    set_style()
    plt.figure(figsize=(10, 6))

    production, ablation, transfer, skipped = [], [], [], []

    for run_path in sorted(RESULTS_DIR.glob("run_*")):
        if not run_path.is_dir():
            continue
        name = run_path.name[len("run_"):]
        df = read_curve(run_log(name))
        if df is None:
            skipped.append(name)
            continue
        if name.startswith(ABLATION_PREFIX):
            ablation.append((name, df))
        elif name.startswith("transfer"):
            transfer.append((name, df))
        elif name.startswith("PPO_seed"):
            production.append((name, df))
        else:
            skipped.append(name)

    for name, df in production:
        plt.plot(df[STEP_COLUMN], df[REWARD_COLUMN],
                 label=RUN_LABELS.get(name, name),
                 color=RUN_COLORS.get(name, "#64748b"),
                 alpha=0.9, linewidth=2)

    for name, df in transfer:
        plt.plot(df[STEP_COLUMN], df[REWARD_COLUMN],
                 label=transfer_label(name), color=transfer_color(name),
                 alpha=0.9, linewidth=2)

    if ablation:
        for _, df in ablation:
            plt.plot(df[STEP_COLUMN], df[REWARD_COLUMN],
                     color=ABLATION_COLOR, alpha=0.18, linewidth=1)
        grid, mean_reward = ablation_mean(ablation)
        plt.plot(grid, mean_reward, color=ABLATION_COLOR, linewidth=2.6,
                 label=f"Ablation grid (mean over {len(ablation)} runs)")

    plt.title("Training Curves: Mean Episode Reward")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Normalized Episode Reward Mean")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plotted = len(production) + len(transfer) + len(ablation)
    print(f"Plot saved to {plot_path} ({plotted} runs)")
    if not production:
        print(f"No production policy curve yet ({run_dir('PPO_seed42')}). "
              "experiments/train_rl.py writes it.")
    if skipped:
        print(f"Runs without a usable curve, not plotted: {', '.join(skipped)}")


if __name__ == "__main__":
    plot_training_curves()
