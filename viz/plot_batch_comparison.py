"""
viz/plot_batch_comparison.py

The batch against the part, in the same visual language as the controller
comparison figure.

A batch is ten parts cut with one tool. The tool is replaced when it breaks or
reaches its wear limit, and the replacement costs 500 reward units against the
200 a finished part is worth. The batch aware policy sees the tool life and
the batch that is left, the other two do not.

Output:
    results/plots/batch_vs_ppo.png
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

# The arms, in the order they are told, with the label of the axis.
ARMS = [
    ("PPO, batch aware", "PPO, batch\naware"),
    ("PPO, per part", "PPO, per\npart"),
    ("Receding-horizon controller, per part", "Receding-\nhorizon"),
]


def main():
    path = EVALUATION_DIR / "batch_results.csv"
    if not path.exists():
        print(f"No batch results at {path}. "
              "Run experiments/batch_experiment.py first.")
        return

    df = pd.read_csv(path)
    rows = {arm: df[df['arm'] == arm].iloc[0] for arm, _ in ARMS
            if len(df[df['arm'] == arm])}
    labels = [label for _, label in ARMS]
    colors = [ARM_COLORS.get(arm, '#64748b') for arm, _ in ARMS]

    def values(column, error=None):
        out = []
        for arm, _ in ARMS:
            if arm not in rows:
                out.append(0.0)
            else:
                out.append(float(rows[arm][column]))
        series = [{'name': 'Batch', 'values': out, 'color': colors}]
        if error:
            series[0]['error'] = [float(rows[arm][error]) if arm in rows else 0.0
                                  for arm, _ in ARMS]
        return series

    def panel(ax, column, title, ylabel, error=None, fmt="{:.1f}"):
        grouped_bar_panel(ax, labels, values(column, error), title, ylabel,
                          fmt=fmt, value_size=11, category_colors=colors)

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    panel(axes[0][0], 'mean_batch_reward', "Batch reward",
          "Mean reward over 30 batches", error='reward_std')
    panel(axes[0][1], 'mean_part_reward', "Reward per part",
          "Mean reward of a part")
    panel(axes[0][2], 'tool_changes', "Tool changes per batch",
          "Tools replaced", fmt="{:.2f}")
    panel(axes[1][0], 'parts_done', "Parts cut", "Parts per batch", fmt="{:.0f}")
    panel(axes[1][1], 'batch_T_total', "Batch time", "s")
    panel(axes[1][2], 'batch_E_total', "Batch energy", "kJ")

    save_figure(
        fig, PLOTS_DIR / "batch_vs_ppo.png",
        title="The batch against the part",
        subtitle=("Ten parts, one tool, a replacement costs 500 reward units "
                  "against the 200 a part is worth"))


ARM_SHORT = {
    "PPO, batch aware": "PPO, batch aware",
    "PPO, per part": "PPO, per part",
    "Receding-horizon controller, per part": "Receding-horizon",
}


if __name__ == "__main__":
    main()
