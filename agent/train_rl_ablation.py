"""
agent/train_rl_ablation.py

Energy weight ablation study.

Trains one PPO policy per energy weight w2 (with w1 = 1 - w2) over several
seeds, then evaluates every policy on the same set of randomized machining
scenarios. The result is the empirical trade-off between throughput and
energy, which is the point of the study: if the energy term in the reward
carries no weight, the operating point should not move.

Outputs:
    results/ablation/ablation_results.csv       one row per weight and seed
    results/ablation/ablation_episodes.csv      one row per weight, seed and scenario
    results/plots/ablation_pareto_frontier.png  MRR against Energy
    results/plots/ablation_line_plots.png       Energy and cycle time against weight
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from agent.evaluate import generate_random_configs, evaluate_policy_on_scenarios
from agent.run_experiments import run_experiments
from stable_baselines3 import PPO
from utils.seeding import set_seed
from viz.plotting_utils import set_style

# Energy weights for the grid. w1 is always 1 - w2.
WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
SEEDS = [42, 43, 44]

def model_tag(w2, seed):
    """Directory name for one ablation policy."""
    return f"ablation_w2_{w2:.1f}_seed{seed}".replace('.', '_')

def build_jobs(weights, seeds, timesteps):
    """One training job per (weight, seed) pair."""
    jobs = []
    for w2 in weights:
        w1 = round(1.0 - w2, 2)
        for seed in seeds:
            tag = model_tag(w2, seed)
            jobs.append({
                "algo": "PPO",
                "seed": seed,
                "steps": timesteps,
                "tag": tag,
                "save_dir": str(PROJECT_ROOT / "models" / tag),
                "env_kwargs": {"w1": w1, "w2": w2},
                "tb_log_name": tag,
            })
    return jobs

def evaluate_ablation_model(model_dir, configs, seed=42):
    """Evaluate one ablation policy over the shared scenario set."""
    model_path = Path(model_dir) / "final_model.zip"
    vec_path = Path(model_dir) / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    set_seed(seed)
    model = PPO.load(str(model_path), device="cpu")
    rows = evaluate_policy_on_scenarios(model, vec_path, configs, seed=seed)
    df = pd.DataFrame(rows)

    return {
        'MRR_mean': df['MRR_mean'].mean(),
        'MRR_std': df['MRR_mean'].std(),
        'Energy_mean': df['Energy_total'].mean(),
        'Energy_std': df['Energy_total'].std(),
        'Time_mean': df['T_total'].mean(),
        'Time_std': df['T_total'].std(),
        'n_mean': df['n_mean'].mean(),
        'f_r_mean': df['f_r_mean'].mean(),
        'Ra_max_mean': df['Ra_max'].mean(),
        'VB_max_mean': df['VB_max'].mean(),
        'Ra_viol_rate': df['Ra_violation'].mean(),
        'VB_viol_rate': df['VB_violation'].mean(),
        'Success_rate': df['Success'].mean(),
    }

def run_ablation(weights=WEIGHTS, seeds=SEEDS, timesteps=200_000,
                 num_scenarios=50, eval_seed=42, parallel_slots=8, skip_training=False):
    logs_dir = PROJECT_ROOT / "results" / "ablation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ENERGY WEIGHT ABLATION: THROUGHPUT / ENERGY TRADE-OFF")
    print(f"  Weights w2: {weights} | Seeds: {seeds} | Steps per model: {timesteps:,}")
    print("=" * 70)

    if not skip_training:
        jobs = build_jobs(weights, seeds, timesteps)
        run_experiments(jobs=jobs, parallel_slots=parallel_slots)

    # Every policy is evaluated on the same scenarios, so the rows compare
    # policies rather than scenarios.
    configs = generate_random_configs(num_scenarios, seed=eval_seed)

    records = []
    episode_rows = []

    for w2 in weights:
        w1 = round(1.0 - w2, 2)
        for seed in seeds:
            tag = model_tag(w2, seed)
            model_dir = PROJECT_ROOT / "models" / tag
            print(f"\n[+] Evaluating {tag} ...")

            eval_metrics = evaluate_ablation_model(model_dir, configs, seed=eval_seed)
            records.append({'w1': w1, 'w2': w2, 'seed': seed, **eval_metrics})

            for name, env_kwargs in configs.items():
                episode_rows.append({'w1': w1, 'w2': w2, 'seed': seed, 'config': name, **env_kwargs})

            print(f"    -> MRR: {eval_metrics['MRR_mean']:.1f} mm3/min | "
                  f"Energy: {eval_metrics['Energy_mean']:.1f} kJ | "
                  f"n: {eval_metrics['n_mean']:.0f} RPM | "
                  f"f_r: {eval_metrics['f_r_mean']:.3f} mm/r | "
                  f"Ra viol: {eval_metrics['Ra_viol_rate']*100:.1f}%")

    df_res = pd.DataFrame(records)
    csv_out = logs_dir / "ablation_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved ablation data to {csv_out}")

    # Per-seed rows, aggregated over seeds for the plots
    agg = df_res.groupby('w2').agg(
        MRR_mean=('MRR_mean', 'mean'),
        MRR_std=('MRR_mean', 'std'),
        Energy_mean=('Energy_mean', 'mean'),
        Energy_std=('Energy_mean', 'std'),
        Time_mean=('Time_mean', 'mean'),
        Time_std=('Time_mean', 'std'),
        n_mean=('n_mean', 'mean'),
        f_r_mean=('f_r_mean', 'mean'),
        Success_rate=('Success_rate', 'mean'),
        seeds=('seed', 'count'),
    ).reset_index()
    agg.to_csv(logs_dir / "ablation_summary.csv", index=False)

    plot_ablation(agg, weights, plots_dir)
    return df_res

def plot_ablation(agg, weights, plots_dir):
    """Pareto frontier of MRR against Energy, plus the two weight sweeps."""
    set_style()

    # Pareto frontier: throughput against energy
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(
        agg['Energy_mean'], agg['MRR_mean'],
        xerr=agg['Energy_std'].fillna(0.0), yerr=agg['MRR_std'].fillna(0.0),
        fmt='o', color='#0d9488', ecolor='#0d9488', elinewidth=1.4,
        capsize=4, markersize=9, linewidth=2.0, label='PPO (mean over seeds)'
    )
    for _, row in agg.iterrows():
        ax.annotate(f"w2={row['w2']:.1f}", (row['Energy_mean'], row['MRR_mean']),
                    textcoords="offset points", xytext=(8, -4), fontsize=9, color='#334155')
    ax.set_title("Pareto Frontier: Throughput against Energy", fontweight='bold')
    ax.set_xlabel("Total Energy per Part (kJ)")
    ax.set_ylabel("Mean Material Removal Rate (mm3/min)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    pareto_file = plots_dir / "ablation_pareto_frontier.png"
    plt.savefig(pareto_file, dpi=300)
    plt.close()
    print(f"[+] Saved Pareto frontier to {pareto_file}")

    # Energy and cycle time against the weight
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(agg['w2'], agg['Energy_mean'], '-o', color='#0d9488', linewidth=2.5, markersize=8)
    ax1.fill_between(agg['w2'],
                     agg['Energy_mean'] - agg['Energy_std'].fillna(0.0),
                     agg['Energy_mean'] + agg['Energy_std'].fillna(0.0),
                     color='#0d9488', alpha=0.2)
    ax1.set_title("Energy Consumption vs. Energy Weight", fontweight='bold')
    ax1.set_xlabel("Energy Weight (w2)")
    ax1.set_ylabel("Total Energy per Part (kJ)")
    ax1.set_xticks(weights)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(agg['w2'], agg['Time_mean'], '-o', color='#ea580c', linewidth=2.5, markersize=8)
    ax2.fill_between(agg['w2'],
                     agg['Time_mean'] - agg['Time_std'].fillna(0.0),
                     agg['Time_mean'] + agg['Time_std'].fillna(0.0),
                     color='#ea580c', alpha=0.2)
    ax2.set_title("Machining Cycle Time vs. Energy Weight", fontweight='bold')
    ax2.set_xlabel("Energy Weight (w2)")
    ax2.set_ylabel("Total Cycle Time (s)")
    ax2.set_xticks(weights)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    line_file = plots_dir / "ablation_line_plots.png"
    plt.savefig(line_file, dpi=300)
    plt.close()
    print(f"[+] Saved Energy vs Time line plots to {line_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--steps", type=int, default=200_000, help="Timesteps per model")
    parser.add_argument("--scenarios", type=int, default=50, help="Scenarios per policy")
    parser.add_argument("--slots", type=int, default=8, help="Parallel training slots")
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training, evaluate existing models")
    args = parser.parse_args()

    run_ablation(timesteps=args.steps, num_scenarios=args.scenarios,
                 parallel_slots=args.slots, skip_training=args.evaluate_only)
