"""
viz/plot_reward_sensitivity.py

How much of the comparison against the classical controller is decided by the
reward, in the same visual language as the controller comparison figure.

The reward credits the removal rate of every pass, so the total credit grows
with the number of passes a plan takes. The script shows what that is worth:
the reward of both arms as the objective scale is swept, under the accounting
the environment uses and under the part level alternative that removes the
pass count from it.

Output:
    results/plots/reward_sensitivity.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.config import EVALUATION_DIR, PLOTS_DIR
from viz.plotting_utils import (
    ARM_COLORS,
    save_figure,
    set_style,
)

CONTROLLER = "Receding-horizon controller"
POLICY = "PPO"
FORMS = [
    ("the sum of the per-pass rates", "The reward as the environment pays it"),
    ("the material of the part against the energy of the part",
     "The part level alternative"),
]


def main():
    path = EVALUATION_DIR / "reward_sensitivity.csv"
    if not path.exists():
        print(f"No reward sensitivity at {path}. "
              "Run experiments/reward_sensitivity.py first.")
        return

    df = pd.read_csv(path)
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, (form, title) in zip(axes[:2], FORMS):
        part = df[df['objective_form'] == form]
        pivot = part.pivot(index='objective_scale', columns='arm',
                           values='mean_reward')
        x = pivot.index
        ax.plot(x, pivot[CONTROLLER], marker='o', color=ARM_COLORS[CONTROLLER],
                linewidth=2.4, label="Receding-horizon")
        ax.plot(x, pivot[POLICY], marker='s', color=ARM_COLORS[POLICY],
                linewidth=2.4, label="PPO")
        ax.fill_between(x, pivot[CONTROLLER], pivot[POLICY],
                        where=(pivot[POLICY] >= pivot[CONTROLLER]),
                        color=ARM_COLORS[POLICY], alpha=0.13, interpolate=True,
                        label="the learned policy is ahead")
        ax.axvline(100, color="#334155", linestyle=":", linewidth=1.4,
                   label="the scale of the configuration")
        ax.set_xscale("log")
        ax.set_xticks(sorted(x))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Objective scale, the weight of the throughput and energy term")
        ax.set_ylabel("Mean episode reward over 100 scenarios")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", framealpha=0.92, fontsize=9)

    # The gap between the two arms, where the sign is the whole story.
    ax = axes[2]
    for index, (form, label) in enumerate(FORMS):
        part = df[df['objective_form'] == form]
        pivot = part.pivot(index='objective_scale', columns='arm',
                           values='mean_reward')
        gap = pivot[POLICY] - pivot[CONTROLLER]
        ax.plot(gap.index, gap, marker='o', linewidth=2.4,
                color=ARM_COLORS[POLICY] if index == 0 else "#0d9488",
                label=label)
    ax.axhline(0, color="#334155", linewidth=1.4)
    ax.axvline(100, color="#334155", linestyle=":", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xticks(sorted(df['objective_scale'].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("The gap between the arms", fontweight="bold")
    ax.set_xlabel("Objective scale")
    ax.set_ylabel("PPO minus the receding-horizon controller")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best", framealpha=0.92, fontsize=9)

    save_figure(
        fig, PLOTS_DIR / "reward_sensitivity.png",
        title="How much of the comparison is decided by the reward",
        subtitle=("The same 100 scenarios, the same energy weight, the reward "
                  "accounting varied and the scale swept"))


if __name__ == "__main__":
    main()
