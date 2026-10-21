"""
viz/plot_robustness.py

The three regimes of the comparison against the classical controller, in the
same visual language as the controller comparison figure.

The regimes are the nominal machine, a machine whose material the controller
has no model for, and a machine whose wear measurement is noisy. Every arm is
measured on the same 100 randomized scenarios at the same energy weight.

Output:
    results/plots/robustness_vs_controller.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.config import EVALUATION_DIR, PLOTS_DIR
from viz.plotting_utils import (
    ARM_COLORS,
    grouped_bar_panel,
    save_figure,
    set_style,
)

# The regimes, in the order they are told, with the label of the axis.
REGIMES = [
    ("nominal", "Nominal\nmachine"),
    ("model error", "Wrong model\nof the material"),
    ("noisy wear measurement", "Noisy wear\nmeasurement"),
]

CONTROLLER = "Receding-horizon controller"
POLICY = "PPO"


def load():
    path = EVALUATION_DIR / "robustness_results.csv"
    if not path.exists():
        print(f"No robustness results at {path}. "
              "Run experiments/robustness_experiment.py first.")
        return None
    return pd.read_csv(path)


def regime_rows(df, regime, arm):
    """The rows of one arm in one regime."""
    part = df[(df['regime'] == regime) & (df['arm'] == arm)]
    return part.iloc[0] if len(part) else None


def main():
    df = load()
    if df is None:
        return

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    labels = [label for _, label in REGIMES]

    def series(column, error=None):
        controller, policy = [], []
        controller_err, policy_err = [], []
        for regime, _ in REGIMES:
            row_c = regime_rows(df, regime, CONTROLLER)
            row_p = regime_rows(df, regime, POLICY)
            if row_c is None or row_p is None:
                continue
            controller.append(row_c[column])
            policy.append(row_p[column])
            if error:
                controller_err.append(row_c[error])
                policy_err.append(row_p[error])
        out = [
            {'name': 'Receding-horizon', 'values': controller,
             'color': ARM_COLORS[CONTROLLER]},
            {'name': 'PPO', 'values': policy, 'color': ARM_COLORS[POLICY]},
        ]
        if error:
            out[0]['error'] = controller_err
            out[1]['error'] = policy_err
        return out

    grouped_bar_panel(axes[0][0], labels, series('mean_reward', 'reward_std'),
                      "Episode reward (the objective)",
                      "Mean reward over 100 scenarios", ylim=(-40, 380))
    grouped_bar_panel(axes[0][1], labels, series('reward_std'),
                      "Reward spread across the scenarios",
                      "Std of the episode reward")
    grouped_bar_panel(axes[0][2], labels, series('MRR_mean', 'MRR_std'),
                      "Material removal rate", "MRR (mm3/min)")
    grouped_bar_panel(axes[1][0], labels, series('Energy_per_part'),
                      "Energy per part", "kJ")
    grouped_bar_panel(axes[1][1], labels, series('Cycle_time'),
                      "Cycle time per part", "s")
    grouped_bar_panel(axes[1][2], labels, series('Success'),
                      "Parts completed within spec", "Share of the scenarios (%)",
                      fmt="{:.0f}", ylim=(0, 118))

    # The breakage, on the same axis as the completion, as a second series.
    ax = axes[1][2]
    ax.clear()
    completed = [regime_rows(df, regime, CONTROLLER)['Success'] * 100
                 for regime, _ in REGIMES]
    broken_c = [regime_rows(df, regime, CONTROLLER)['Broken'] * 100
                for regime, _ in REGIMES]
    completed_p = [regime_rows(df, regime, POLICY)['Success'] * 100
                   for regime, _ in REGIMES]
    broken_p = [regime_rows(df, regime, POLICY)['Broken'] * 100
                for regime, _ in REGIMES]
    positions = np.arange(len(REGIMES))
    width = 0.19
    ax.bar(positions - 1.5 * width, completed, width, color="#0d9488",
           label="In spec, controller", zorder=3)
    ax.bar(positions - 0.5 * width, broken_c, width, color="#b91c1c",
           label="Broken, controller", zorder=3)
    ax.bar(positions + 0.5 * width, completed_p, width, color="#1f77b4",
           label="In spec, PPO", zorder=3)
    ax.bar(positions + 1.5 * width, broken_p, width, color="#f59e0b",
           label="Broken, PPO", zorder=3)
    for pos, value in zip(positions - 1.5 * width, completed):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#334155",
                    fontweight="bold")
    for pos, value in zip(positions + 0.5 * width, completed):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#334155",
                    fontweight="bold")
    for pos, value in zip(positions - 0.5 * width, broken_c):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#334155",
                    fontweight="bold")
    for pos, value in zip(positions + 1.5 * width, broken_p):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#334155",
                    fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Reliability over 100 scenarios", fontweight="bold")
    ax.set_ylabel("Share of the scenarios (%)")
    ax.set_ylim(0, 132)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8,
              framealpha=0.95)

    save_figure(
        fig, PLOTS_DIR / "robustness_vs_controller.png",
        title="Where the learned policy overtakes the classical controller",
        subtitle=("Three regimes, the same 100 randomized scenarios, the same "
                  "energy weight w2 = 0.4"))


if __name__ == "__main__":
    main()
