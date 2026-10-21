"""
experiments/summarize_results.py

Builds the result tables quoted in README.md and in Part 3 of PROJECT.md
straight from the committed artifacts, so the documentation cannot drift away
from the runs that produced it.

Reads:
    results/run_PPO_seed42/metrics/monte_carlo.csv
    results/run_PPO_seed42/metrics/weight_sweep.csv
    results/evaluation/ablation_summary.csv
    results/evaluation/baselines.csv
    results/evaluation/controller_init_results.csv
    results/evaluation/transfer_results.csv

Writes:
    results/RESULTS.md
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate import compute_statistics
from MillingDT.config import (
    EVALUATION_DIR,
    PRIMARY_RUN,
    RESULTS_DIR,
    run_metrics,
)

def monte_carlo_table():
    """Per-scenario statistics of the primary policy."""
    path = run_metrics(PRIMARY_RUN) / "monte_carlo.csv"
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    stats_dict = compute_statistics(df)

    rows = [
        ('Material removal rate (mm3/min)', 'MRR_mean', 2),
        ('Energy per part (kJ)', 'Energy_total', 2),
        ('Cycle time per part (s)', 'T_total', 2),
        ('Max surface roughness (um)', 'Ra_max', 3),
        ('Max flank wear (mm)', 'VB_max', 4),
    ]

    lines = [
        "| Metric | Mean | Std | 95% CI |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for label, key, digits in rows:
        s = stats_dict[key]
        lines.append(
            f"| {label} | {s['mean']:.{digits}f} | {s['std']:.{digits}f} | "
            f"[{s['ci_lower']:.{digits}f}, {s['ci_upper']:.{digits}f}] |"
        )

    for label, key in (('Roughness violations, worst pass', 'Ra_violation'),
                       ('Roughness violations, finished surface', 'Ra_final_violation'),
                       ('Roughness above the safety limit, worst pass', 'Ra_safe_violation'),
                       ('Roughness above the safety limit, finished surface', 'Ra_final_safe_violation'),
                       ('Flank wear violations (VB > 0.3 mm)', 'VB_violation'),
                       ('Tool breakage', 'Broken')):
        s = stats_dict[key]
        lines.append(
            f"| {label} | {s['rate']*100:.1f}% | | "
            f"[{s['ci_lower']*100:.1f}%, {s['ci_upper']*100:.1f}%] |"
        )
    lines.append(f"| Parts completed within spec | {df['Success'].mean()*100:.1f}% | | |")
    lines.append(f"| Scenarios | {len(df)} | | |")

    return "\n".join(lines), df

def ablation_table():
    """Energy weight ablation, aggregated over seeds."""
    path = EVALUATION_DIR / "ablation_summary.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).sort_values('w2')

    lines = [
        "| w2 (energy weight) | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Spindle (RPM) | Feed (mm/r) | On the frontier |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for _, r in df.iterrows():
        mrr_std = 0.0 if pd.isna(r['MRR_std']) else r['MRR_std']
        energy_std = 0.0 if pd.isna(r['Energy_std']) else r['Energy_std']
        time_std = 0.0 if pd.isna(r['Time_std']) else r['Time_std']
        on_frontier = "yes" if bool(r.get('pareto', False)) else "dominated"
        lines.append(
            f"| {r['w2']:.1f} | {r['MRR_mean']:.1f} +/- {mrr_std:.1f} | "
            f"{r['Energy_mean']:.1f} +/- {energy_std:.1f} | "
            f"{r['Time_mean']:.1f} +/- {time_std:.1f} | "
            f"{r['n_mean']:.0f} | {r['f_r_mean']:.3f} | {on_frontier} |"
        )
    return "\n".join(lines)

def baseline_table():
    """Classical baselines against the trained policy."""
    path = EVALUATION_DIR / "baselines.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    labels = {
        "constant_high": "Constant high",
        "constant_safe": "Constant safe",
        "rule_based": "Rule based",
        "nsga2": "NSGA-II static",
        "receding_horizon": "Receding-horizon",
    }

    lines = [
        "| Policy | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Mean reward | Ra violation rate |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped = df.groupby("policy")
    for policy, g in grouped:
        lines.append(
            f"| {labels.get(policy, policy)} | {g['mean_mrr_mm3_min'].mean():.1f} | "
            f"{g['total_energy_kJ'].mean():.1f} | {g['cycle_time_s'].mean():.1f} | "
            f"{g['total_reward'].mean():.1f} | {g['ra_violation_rate'].mean()*100:.1f}% |"
        )
    return "\n".join(lines)

def weight_sweep_table():
    """The frontier of the production policy, swept at runtime."""
    out_dir = run_metrics(PRIMARY_RUN)

    blocks = []
    path = out_dir / "weight_sweep.csv"
    if path.exists():
        df = pd.read_csv(path).sort_values('w2')
        lines = [
            "### One policy, asked for every point of the trade-off\n",
            "| w2 (energy weight) | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Spindle (RPM) | Feed (mm/r) | Ra violation rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['w2']:.1f} | {r['MRR_mean']:.1f} | {r['Energy_mean']:.1f} | "
                f"{r['Time_mean']:.1f} | {r['n_mean']:.0f} | {r['f_r_mean']:.3f} | "
                f"{r['Ra_viol_rate']*100:.1f}% |"
            )
        blocks.append("\n".join(lines))

    comparison_path = EVALUATION_DIR / "weight_sweep_vs_ablation.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path).sort_values('w2')
        lines = [
            "### One policy against 21 policies, one per weight\n",
            "| w2 (energy weight) | MRR, one policy | MRR, per-weight | MRR gap | Energy, one policy | Energy, per-weight | Energy gap |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['w2']:.1f} | {r['MRR_mean_sweep']:.1f} | {r['MRR_mean_ablation']:.1f} | "
                f"{r['MRR_gap_pct']:+.1f}% | {r['Energy_mean_sweep']:.1f} | "
                f"{r['Energy_mean_ablation']:.1f} | {r['Energy_gap_pct']:+.1f}% |"
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else None


def transfer_table():
    """Zero shot, fine tuned, from scratch and the production policy on a new material."""
    path = EVALUATION_DIR / "transfer_results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).iloc[0]
    lines = [
        "| Arm | Mean episode reward | Ra violation, worst pass | Ra violation, finished surface | Parts completed |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| Source policy, zero shot | {df['zero_shot_reward']:.1f} | {df['zero_shot_ra_violation']*100:.1f}% | {df['zero_shot_ra_final_violation']*100:.1f}% | {df['zero_shot_success']*100:.1f}% |",
        f"| Source policy, fine tuned | {df['finetuned_reward']:.1f} | {df['finetuned_ra_violation']*100:.1f}% | {df['finetuned_ra_final_violation']*100:.1f}% | {df['finetuned_success']*100:.1f}% |",
        f"| Trained on the target from scratch | {df['from_scratch_reward']:.1f} | {df['from_scratch_ra_violation']*100:.1f}% | {df['from_scratch_ra_final_violation']*100:.1f}% | {df['from_scratch_success']*100:.1f}% |",
    ]
    if 'production_reward' in df:
        lines.append(
            f"| Production policy, zero shot | {df['production_reward']:.1f} | "
            f"{df['production_ra_violation']*100:.1f}% | {df['production_ra_final_violation']*100:.1f}% | "
            f"{df['production_success']*100:.1f}% |"
        )
    lines += [
        "",
        f"Source policy: {int(df['source_steps']):,} steps on the nominal material. "
        f"Adaptation budget: {int(df['adapt_budget']):,} steps on the target.",
    ]
    return "\n".join(lines)


def controller_init_table():
    """The controller, the policy trained from scratch and the one started from it."""
    path = EVALUATION_DIR / "controller_init_results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    lines = [
        "| Arm | Mean episode reward | Reward std | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Ra violation, finished surface | Parts completed |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['arm']} | {r['mean_reward']:.1f} | {r['reward_std']:.1f} | "
            f"{r['MRR_mean']:.1f} | {r['Energy_per_part']:.1f} | {r['Cycle_time']:.1f} | "
            f"{r['Ra_violation_finished_surface']*100:.1f}% | {r['Success']*100:.1f}% |"
        )
    lines += [
        "",
        "All three arms are measured on the same 100 randomized scenarios at the "
        "same energy weight. The reward std is the spread of the episode reward "
        "over those scenarios.",
    ]
    return "\n".join(lines)


def figures_table():
    """The figures, and what each one shows."""
    figures = [
        ("results/plots/controller_vs_ppo.png",
         "The receding-horizon controller against the two learned policies, "
         "over the same 100 scenarios"),
        ("results/plots/baseline_vs_ppo_comparison.png",
         "The six policies side by side: removal rate, energy per part and the reward"),
        ("results/plots/ablation_pareto_frontier.png",
         "The ablation frontier: removal rate against energy, the non-dominated points joined"),
        ("results/plots/ablation_line_plots.png",
         "Energy per part and cycle time against the energy weight"),
        ("results/plots/training_curves.png",
         "The training curve of every run: the production policy, the transfer arms "
         "and the ablation grid"),
        ("results/run_PPO_seed42/plots/monte_carlo_failure_landscape.png",
         "The operating envelope of the production policy: flank wear against "
         "roughness over the 100 scenarios"),
        ("results/run_PPO_seed42/plots/monte_carlo_constraints_kde_ra.png",
         "The distribution of the worst-pass surface roughness over the 100 scenarios"),
        ("results/run_PPO_seed42/plots/monte_carlo_constraints_kde_vb.png",
         "The distribution of the flank wear over the 100 scenarios"),
    ]

    lines = [
        "| Figure | What it shows |",
        "| :--- | :--- |",
    ]
    for path, description in figures:
        lines.append(f"| `{path}` | {description} |")
    return "\n".join(lines)


def main():
    parts = ["# Measured Results\n",
             "Generated by `python -m experiments.summarize_results`. Every figure below comes "
             "from the artifact named in the section heading.\n"]

    mc, df = monte_carlo_table()
    if mc:
        parts.append("## Production policy, Monte Carlo sweep\n")
        parts.append("Artifact: `results/run_PPO_seed42/metrics/monte_carlo.csv`\n")
        parts.append(mc + "\n")
        if df is not None:
            parts.append(
                f"Mean operating point across the {len(df)} scenarios: "
                f"{df['n_mean'].mean():.0f} RPM, {df['f_r_mean'].mean():.3f} mm/r, "
                f"{df['a_p_mean'].mean():.2f} mm depth of cut, "
                f"{df['passes'].mean():.1f} passes per part.\n"
            )

    ab = ablation_table()
    if ab:
        parts.append("## Energy weight ablation\n")
        parts.append("Artifact: `results/evaluation/ablation_summary.csv`\n")
        parts.append(ab + "\n")

    ws = weight_sweep_table()
    if ws:
        parts.append("## Production policy, weight swept at runtime\n")
        parts.append("Artifact: `results/run_PPO_seed42/metrics/weight_sweep.csv`\n")
        parts.append(ws + "\n")

    bl = baseline_table()
    if bl:
        parts.append("## Classical baselines\n")
        parts.append("Artifact: `results/evaluation/baselines.csv`\n")
        parts.append(bl + "\n")

    ci = controller_init_table()
    if ci:
        parts.append("## Starting the policy from the receding-horizon controller\n")
        parts.append("Artifact: `results/evaluation/controller_init_results.csv`\n")
        parts.append(ci + "\n")

    tr = transfer_table()
    if tr:
        parts.append("## Transfer to a harder material\n")
        parts.append("Artifact: `results/evaluation/transfer_results.csv`\n")
        parts.append(tr + "\n")

    parts.append("## Figures\n")
    parts.append("Regenerated by the scripts in `viz/`.\n")
    parts.append(figures_table() + "\n")

    out = "\n".join(parts)
    out_path = RESULTS_DIR / "RESULTS.md"
    out_path.write_text(out, encoding="utf-8")
    print(out)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
