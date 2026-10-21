"""
viz/plot_robustness.py

Every setup error the study measures, in the same visual language as the
controller comparison figure.

The cases are the two ways the setup's model of the machine can be wrong: the
wear measurement is noisy, or the material model is miscalibrated. Every arm
is measured on the same 100 randomized scenarios at the same energy weight,
and both arms get the same wrong information.

The data comes from the regime map, which sweeps both errors and writes one
row per arm, per case, per metric.

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

CONTROLLER = "Receding-horizon controller"
POLICY = "PPO"

# The cases, in the order they are told, with the label of the axis and the
# row of the regime map they come from.
CASES = [
    (("measurement noise", 0.0), "None"),
    (("measurement noise", 0.02), "Wear noise\n2%"),
    (("measurement noise", 0.10), "Wear noise\n10%"),
    (("model error", 0.05), "Model off\n5%"),
    (("model error", 0.20), "Model off\n20%"),
    (("model error", 0.30), "Model off\n30%"),
]


def load():
    path = EVALUATION_DIR / "regime_map.csv"
    if not path.exists():
        print(f"No regime map at {path}. Run experiments/regime_map.py first.")
        return None
    return pd.read_csv(path)


def case_row(df, sweep, level, arm):
    """The row of one arm in one case."""
    column = 'noise' if sweep == 'measurement noise' else 'model_error'
    part = df[(df['sweep'] == sweep)
              & (np.isclose(df[column], level))
              & (df['arm'] == arm)]
    return part.iloc[0] if len(part) else None


def main():
    df = load()
    if df is None:
        return

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    labels = [label for _, label in CASES]

    def series(column, error=None):
        controller, policy = [], []
        controller_err, policy_err = [], []
        for (sweep, level), _ in CASES:
            row_c = case_row(df, sweep, level, CONTROLLER)
            row_p = case_row(df, sweep, level, POLICY)
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
                      "Mean reward over 100 scenarios", ylim=(-40, 380),
                      label_size=10, value_size=8)
    grouped_bar_panel(axes[0][1], labels, series('reward_std'),
                      "Reward spread across the scenarios",
                      "Std of the episode reward", label_size=10, value_size=8)
    grouped_bar_panel(axes[0][2], labels, series('MRR_mean', 'MRR_std'),
                      "Material removal rate", "MRR (mm3/min)",
                      label_size=10, value_size=8)
    grouped_bar_panel(axes[1][0], labels, series('Energy_per_part'),
                      "Energy per part", "kJ", label_size=10, value_size=8)
    grouped_bar_panel(axes[1][1], labels, series('Cycle_time'),
                      "Cycle time per part", "s", label_size=10, value_size=8)
    grouped_bar_panel(axes[1][2], labels, series('Success'),
                      "Parts completed within spec", "Share of the scenarios (%)",
                      fmt="{:.0f}", ylim=(0, 118), label_size=10, value_size=8)

    # The breakage, on the same axis as the completion, as a second series.
    ax = axes[1][2]
    ax.clear()
    completed_c = [case_row(df, sweep, level, CONTROLLER)['Success'] * 100
                   for (sweep, level), _ in CASES]
    broken_c = [case_row(df, sweep, level, CONTROLLER)['Broken'] * 100
                for (sweep, level), _ in CASES]
    completed_p = [case_row(df, sweep, level, POLICY)['Success'] * 100
                   for (sweep, level), _ in CASES]
    broken_p = [case_row(df, sweep, level, POLICY)['Broken'] * 100
                for (sweep, level), _ in CASES]
    positions = np.arange(len(CASES))
    width = 0.19
    ax.bar(positions - 1.5 * width, completed_c, width, color="#0d9488",
           label="In spec, controller", zorder=3)
    ax.bar(positions - 0.5 * width, broken_c, width, color="#b91c1c",
           label="Broken, controller", zorder=3)
    ax.bar(positions + 0.5 * width, completed_p, width, color="#1f77b4",
           label="In spec, PPO", zorder=3)
    ax.bar(positions + 1.5 * width, broken_p, width, color="#f59e0b",
           label="Broken, PPO", zorder=3)
    for pos, value in zip(positions - 1.5 * width, completed_c):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#334155",
                    fontweight="bold")
    for pos, value in zip(positions + 0.5 * width, completed_p):
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
    ax.set_xticklabels(labels, fontsize=9)
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
        subtitle=("Every way the setup's model of the machine can be wrong, the "
                  "same 100 randomized scenarios, the same energy weight w2 = 0.4"))


if __name__ == "__main__":
    main()
