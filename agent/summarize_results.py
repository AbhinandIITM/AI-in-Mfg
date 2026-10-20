"""
agent/summarize_results.py

Builds the result tables quoted in README.md and in Part 3 of PROJECT.md
straight from the committed artifacts, so the documentation cannot drift away
from the runs that produced it.

Reads:
    results/monte_carlo/eval_PPO_MC_v5.csv
    results/ablation/ablation_summary.csv
    results/baseline_results_v5.csv

Writes:
    results/RESULTS.md
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluate import compute_statistics

def monte_carlo_table():
    """Per-scenario statistics of the primary policy."""
    path = PROJECT_ROOT / "results" / "monte_carlo" / "eval_PPO_MC_v5.csv"
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
    path = PROJECT_ROOT / "results" / "ablation" / "ablation_summary.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).sort_values('w2')

    lines = [
        "| w2 (energy weight) | MRR (mm3/min) | Energy per part (kJ) | Energy per pass (kJ) | Cycle time (s) | Spindle (RPM) | Feed (mm/r) | Depth of cut (mm) | Passes | Seeds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        mrr_std = 0.0 if pd.isna(r['MRR_std']) else r['MRR_std']
        energy_std = 0.0 if pd.isna(r['Energy_std']) else r['Energy_std']
        time_std = 0.0 if pd.isna(r['Time_std']) else r['Time_std']
        lines.append(
            f"| {r['w2']:.1f} | {r['MRR_mean']:.1f} +/- {mrr_std:.1f} | "
            f"{r['Energy_mean']:.1f} +/- {energy_std:.1f} | "
            f"{r['Energy_per_pass']:.2f} | "
            f"{r['Time_mean']:.1f} +/- {time_std:.1f} | "
            f"{r['n_mean']:.0f} | {r['f_r_mean']:.3f} | {r['a_p_mean']:.2f} | "
            f"{r['passes_mean']:.1f} | {int(r['seeds'])} |"
        )
    return "\n".join(lines)

def baseline_table():
    """Classical baselines against the trained policy."""
    path = PROJECT_ROOT / "results" / "baseline_results_v5.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    labels = {
        "constant_high": "Constant high",
        "constant_safe": "Constant safe",
        "rule_based": "Rule based",
        "nsga2": "NSGA-II static",
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

def conditioned_table():
    """The frontier of the conditioned policy, at both training budgets."""
    out_dir = PROJECT_ROOT / "results" / "conditioned"

    blocks = []
    for filename, title in (("conditioned_frontier.csv",
                             "One conditioned policy, 1,000,000 steps"),
                            ("conditioned_frontier_150k.csv",
                             "One conditioned policy, 150,000 steps, the same budget as the sweep")):
        path = out_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values('w2')
        lines = [
            f"### {title}\n",
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

    return "\n\n".join(blocks) if blocks else None


def transfer_table():
    """Zero shot, fine tuned and from scratch on a new material."""
    path = PROJECT_ROOT / "results" / "transfer" / "transfer_results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).iloc[0]
    lines = [
        "| Arm | Mean episode reward | Ra violation rate | Parts completed |",
        "| :--- | ---: | ---: | ---: |",
        f"| Source policy, zero shot | {df['zero_shot_reward']:.1f} | {df['zero_shot_ra_violation']*100:.1f}% | {df['zero_shot_success']*100:.1f}% |",
        f"| Source policy, fine tuned | {df['finetuned_reward']:.1f} | {df['finetuned_ra_violation']*100:.1f}% | {df['finetuned_success']*100:.1f}% |",
        f"| Trained on the target from scratch | {df['from_scratch_reward']:.1f} | {df['from_scratch_ra_violation']*100:.1f}% | {df['from_scratch_success']*100:.1f}% |",
        "",
        f"Source policy: {int(df['source_steps']):,} steps on the nominal material. "
        f"Adaptation budget: {int(df['adapt_budget']):,} steps on the target.",
    ]
    return "\n".join(lines)


def main():
    parts = ["# Measured Results\n",
             "Generated by `python agent/summarize_results.py`. Every figure below comes "
             "from the artifact named in the section heading.\n"]

    mc, df = monte_carlo_table()
    if mc:
        parts.append("## Primary PPO policy, Monte Carlo sweep\n")
        parts.append("Artifact: `results/monte_carlo/eval_PPO_MC_v5.csv`\n")
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
        parts.append("Artifact: `results/ablation/ablation_summary.csv`\n")
        parts.append(ab + "\n")

    cond = conditioned_table()
    if cond:
        parts.append("## Conditioned policy, weight swept at runtime\n")
        parts.append("Artifacts: `results/conditioned/conditioned_frontier.csv` and "
                     "`results/conditioned/conditioned_frontier_150k.csv`\n")
        parts.append(cond + "\n")

    bl = baseline_table()
    if bl:
        parts.append("## Classical baselines\n")
        parts.append("Artifact: `results/baseline_results_v5.csv`\n")
        parts.append(bl + "\n")

    tr = transfer_table()
    if tr:
        parts.append("## Transfer to a harder material\n")
        parts.append("Artifact: `results/transfer/transfer_results.csv`\n")
        parts.append(tr + "\n")

    out = "\n".join(parts)
    out_path = PROJECT_ROOT / "results" / "RESULTS.md"
    out_path.write_text(out, encoding="utf-8")
    print(out)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
