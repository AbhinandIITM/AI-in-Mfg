"""
viz/plot_controller_comparison.py

Compares the receding-horizon controller against the two learned policies.

Three arms, measured on the same 100 randomized scenarios at the same energy
weight:

    the receding-horizon controller,
    the production policy, trained from scratch with PPO,
    a policy whose actor was initialized from the controller.

The figure answers two questions at once. On the objective the three arms are
level, and the spread of the episode reward across the scenarios is what
separates them. On the physical metrics the controller is ahead on removal
rate and cycle time, and the learned policies are ahead on reliability.

Output:
    results/plots/controller_vs_ppo.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.config import EVALUATION_DIR, PLOTS_DIR
from viz.plotting_utils import set_style

# One colour per arm, reused in every panel.
ARM_COLORS = {
    "Receding-horizon controller": "#64748b",
    "PPO from scratch": "#1f77b4",
    "PPO initialized from the controller": "#ea580c",
}
ARM_SHORT = {
    "Receding-horizon controller": "Controller",
    "PPO from scratch": "PPO\nfrom scratch",
    "PPO initialized from the controller": "PPO from\ncontroller",
}


def load_results():
    """The three arms, from the artifact of the controller experiment."""
    path = EVALUATION_DIR / "controller_init_results.csv"
    if not path.exists():
        print(f"No controller initialization results at {path}. "
              "Run experiments/controller_experiment.py first.")
        return None
    return pd.read_csv(path)


def bar_panel(ax, df, column, title, ylabel, error=None, colors=None, fmt="{:.1f}"):
    """One bar chart, one metric, with the arm colours."""
    arms = list(df["arm"])
    values = df[column].to_numpy(dtype=float)
    bar_colors = colors or [ARM_COLORS.get(arm, "#64748b") for arm in arms]

    bars = ax.bar(range(len(arms)), values, color=bar_colors, width=0.62,
                  zorder=3, edgecolor="white")
    spread = df[error].to_numpy(dtype=float) if error else np.zeros(len(arms))
    if error:
        ax.errorbar(range(len(arms)), values, yerr=spread,
                    fmt="none", ecolor="#334155", elinewidth=1.4, capsize=4, zorder=4)

    # The value sits above the error bar, so the two never overprint.
    for bar, value, top in zip(bars, values, values + spread):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, top),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=10, color="#334155", fontweight="bold")

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_SHORT.get(arm, arm) for arm in arms], fontsize=10)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def plot_controller_comparison():
    df = load_results()
    if df is None:
        return

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    bar_panel(axes[0][0], df, "mean_reward",
              "Episode reward (the objective)", "Mean reward over 100 scenarios",
              error="reward_std")
    bar_panel(axes[0][1], df, "reward_std",
              "Reward spread across the scenarios", "Std of the episode reward")
    bar_panel(axes[0][2], df, "MRR_mean",
              "Material removal rate", "MRR (mm3/min)", error="MRR_std")
    bar_panel(axes[1][0], df, "Energy_per_part",
              "Energy per part", "kJ")
    bar_panel(axes[1][1], df, "Cycle_time",
              "Cycle time per part", "s")

    # Reliability: the parts completed and the tools broken.
    ax = axes[1][2]
    arms = list(df["arm"])
    width = 0.38
    positions = np.arange(len(arms))
    completed = df["Success"].to_numpy(dtype=float) * 100
    broken = df["Broken"].to_numpy(dtype=float) * 100
    ax.bar(positions - width / 2, completed, width, label="Parts completed in spec",
           color="#0d9488", zorder=3)
    ax.bar(positions + width / 2, broken, width, label="Tools broken",
           color="#b91c1c", zorder=3)
    for pos, value in zip(positions - width / 2, completed):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10, color="#334155",
                    fontweight="bold")
    for pos, value in zip(positions + width / 2, broken):
        ax.annotate(f"{value:.0f}%", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10, color="#334155",
                    fontweight="bold")
    # The two series are named on the bars themselves, so the panel needs no
    # legend and nothing overlaps the small breakage bars.
    ax.annotate("in spec", (positions[0] - width / 2, completed[0]),
                textcoords="offset points", xytext=(0, 20), ha="center",
                fontsize=9, color="#0d9488", fontweight="bold")
    ax.annotate("broken", (positions[0] + width / 2, broken[0]),
                textcoords="offset points", xytext=(0, 20), ha="center",
                fontsize=9, color="#b91c1c", fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([ARM_SHORT.get(arm, arm) for arm in arms], fontsize=10)
    ax.set_title("Reliability over 100 scenarios", fontweight="bold")
    ax.set_ylabel("Share of the scenarios (%)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Receding-horizon controller against the learned policies\n"
        "Same 100 randomized scenarios, same energy weight w2 = 0.4",
        fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_plot = PLOTS_DIR / "controller_vs_ppo.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Controller comparison saved to {out_plot}")

    print("\nWhat the figure says:")
    print(f"  mean reward: controller {df.loc[0, 'mean_reward']:.1f}, "
          f"PPO {df.loc[1, 'mean_reward']:.1f}, "
          f"PPO from controller {df.loc[2, 'mean_reward']:.1f}")
    print(f"  reward std:  controller {df.loc[0, 'reward_std']:.1f}, "
          f"PPO {df.loc[1, 'reward_std']:.1f}, "
          f"PPO from controller {df.loc[2, 'reward_std']:.1f}")
    print(f"  MRR:         controller {df.loc[0, 'MRR_mean']:.1f}, "
          f"PPO {df.loc[1, 'MRR_mean']:.1f}, "
          f"PPO from controller {df.loc[2, 'MRR_mean']:.1f} mm3/min")


if __name__ == "__main__":
    plot_controller_comparison()
