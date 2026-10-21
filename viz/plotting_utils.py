"""
viz/plotting_utils.py

Refactored plotting functions for the CNC Milling project.
Follows the DRY principle to reduce code duplication between plot_eval.py and plot_results.py.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def set_style():
    """Set global plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })


# The colours of the arms, shared by every figure that compares them.
ARM_COLORS = {
    "Receding-horizon controller": "#64748b",
    "Receding-horizon, per part": "#64748b",
    "PPO": "#1f77b4",
    "PPO, per part": "#1f77b4",
    "PPO from scratch": "#1f77b4",
    "PPO from controller": "#ea580c",
    "PPO trained for the noise": "#0d9488",
    "PPO, batch aware": "#0d9488",
    "PPO, batch aware ": "#0d9488",
}


def grouped_bar_panel(ax, categories, series, title, ylabel, fmt="{:.1f}",
                      label_size=10, value_size=10, ylim=None, legend=False,
                      category_colors=None):
    """One panel of grouped bars, one group per category.

    Every series is a dict with a name, the values, the colour and an
    optional error to draw as a whisker. The value is written above the bar,
    and above the whisker when there is one, so the two never overprint.

    With one series and a category_colors list, every bar takes the colour
    of its own category, which is what a comparison of arms needs.
    """
    n_series = len(series)
    group_width = 0.78
    bar_width = group_width / max(n_series, 1)
    positions = np.arange(len(categories))

    for index, item in enumerate(series):
        values = np.asarray(item['values'], dtype=float)
        errors = item.get('error')
        offset = positions - group_width / 2 + bar_width * (index + 0.5)
        if category_colors is not None:
            colors = [category_colors[i % len(category_colors)]
                      for i in range(len(values))]
        else:
            colors = item.get('color', '#1f77b4')
        bars = ax.bar(offset, values, width=bar_width * 0.92,
                      color=colors, label=item.get('name'),
                      zorder=3, edgecolor="white")
        if errors is not None:
            errors = np.asarray(errors, dtype=float)
            ax.errorbar(offset, values, yerr=errors, fmt="none", ecolor="#334155",
                        elinewidth=1.4, capsize=3, zorder=4)
            tops = values + errors
        else:
            tops = values
        # The label carries the value, and it sits above the whisker when
        # there is one, so the two never overprint.
        for bar, value, top in zip(bars, values, tops):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, top),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=value_size, color="#334155", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=label_size)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="best", framealpha=0.92)


def save_figure(fig, path, title=None, subtitle=None):
    """Save a figure with the project's title block and clean up."""
    if title:
        heading = title if not subtitle else f"{title}\n{subtitle}"
        fig.suptitle(heading, fontsize=15, fontweight="bold", y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

def plot_kde_distribution(data, title, xlabel, ylabel="Density", color="blue", limit=None, limit_label=None, safety_limit=None, safety_limit_label=None, filename=None):
    """
    Plot KDE distribution with mean, median, and constraint limit overlays.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(data=data, fill=True, alpha=0.55, color=color, label="Empirical Density")

    mean_val = np.mean(data)
    median_val = np.median(data)

    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=1.8, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color='teal', linestyle=':', linewidth=1.8, label=f"Median: {median_val:.2f}")

    if limit is not None:
        label_txt = limit_label or f"Constraint Limit: {limit:.2f}"
        ax.axvline(limit, color='crimson', linestyle='solid', linewidth=2.5, label=label_txt)
        
    if safety_limit is not None:
        safety_txt = safety_limit_label or f"Safety Limit: {safety_limit:.2f}"
        ax.axvline(safety_limit, color='orange', linestyle='--', linewidth=2.0, label=safety_txt)

        # Ensure both data and constraint limit are clearly visible on x-axis
        d_min = min(float(np.min(data)), 0.0)
        d_max = max(float(np.max(data)), float(limit))
        margin = max((d_max - d_min) * 0.12, 0.02)
        ax.set_xlim(d_min, d_max + margin)

        # Highlight violation zone beyond limit
        ax.axvspan(limit, d_max + margin, color='red', alpha=0.12, label="Violation Zone")

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()

def plot_failure_landscape(df, x_col, y_col, title, x_label=None, y_label=None, 
                           x_limit=None, y_limit=None, x_limit_label=None, y_limit_label=None, x_safe=None, y_safe=None, x_safe_label=None, y_safe_label=None, filename=None):
    """
    Scatter plot of success/failure landscape with engineering constraint boundaries.
    """
    plt.figure(figsize=(12, 8))

    # Define success as neither broken nor exceeding Ra_max
    if 'success' not in df.columns:
        df = df.copy()
        df['success'] = ~(df.get('broken', False) | (df.get('Ra_violation', False)))

    # Determine axis boundaries encompassing data and limits
    max_x = max(float(df[x_col].max()), float(x_limit) if x_limit is not None else 0.0)
    max_y = max(float(df[y_col].max()), float(y_limit) if y_limit is not None else 0.0)
    x_margin = max(max_x * 0.15, 0.03)
    y_margin = max(max_y * 0.15, 0.2)
    x_upper = max_x + x_margin
    y_upper = max_y + y_margin

    # Safe operating window shading
    fill_x = x_safe if x_safe is not None else (x_limit if x_limit is not None else 0)
    fill_y = y_safe if y_safe is not None else (y_limit if y_limit is not None else 0)
    if fill_x and fill_y:
        plt.fill_between([0, fill_x], 0, fill_y, color='green', alpha=0.15, label='Safe Operating Window')
        
    if x_safe is not None:
        plt.axvline(x_safe, color='orange', linestyle='--', linewidth=1.5, label=x_safe_label or f"X Safe ({x_safe})")
    if y_safe is not None:
        plt.axhline(y_safe, color='orange', linestyle='--', linewidth=1.5, label=y_safe_label or f"Y Safe ({y_safe})")

    if x_limit is not None:
        plt.axvline(x_limit, color='crimson', linestyle='--', linewidth=2.0, 
                    label=x_limit_label or f"Wear Limit ({x_col} ≤ {x_limit})")

    if y_limit is not None:
        plt.axhline(y_limit, color='darkorange', linestyle='--', linewidth=2.0, 
                    label=y_limit_label or f"Roughness Limit ({y_col} ≤ {y_limit})")

    # Draw scatter points
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue='success',
        style='success', palette={True: "forestgreen", False: "crimson"},
        markers={True: "o", False: "X"}, s=110, alpha=0.85
    )

    plt.xlim(0, x_upper)
    plt.ylim(0, y_upper)

    plt.title(title, fontweight='bold')
    plt.xlabel(x_label if x_label else x_col)
    plt.ylabel(y_label if y_label else y_col)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()
