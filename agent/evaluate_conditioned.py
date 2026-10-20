"""
agent/evaluate_conditioned.py

Sweeps the energy weight at runtime for the conditioned policy and writes the
resulting frontier, then compares it against the per-weight sweep that
agent/train_rl_ablation.py produced.

The comparison is the test of the claim that one conditioned policy covers the
same frontier as one policy per weight.

Outputs:
    results/conditioned/conditioned_frontier.csv
    results/conditioned/conditioned_vs_sweep.csv
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from agent.evaluate import evaluate_policy_on_scenarios
from agent.train_rl_ablation import WEIGHTS
from config.scenarios import generate_random_configs
from utils.seeding import set_seed

ENV_ID = "MillingDTEnv-w2-v1"


def evaluate_frontier(model_dir, weights=WEIGHTS, num_scenarios=50, seed=42):
    """Run the conditioned policy once per energy weight."""
    model_dir = Path(model_dir)
    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    vec_path = model_dir / "vecnormalize.pkl"

    configs = generate_random_configs(num_scenarios, seed=seed)

    records = []
    for w2 in weights:
        set_seed(seed)
        rows = evaluate_policy_on_scenarios(
            model, vec_path, configs, seed=seed,
            env_id=ENV_ID,
            env_kwargs={"w1": 1.0 - w2, "w2": w2, "randomize_weight": False},
        )
        df = pd.DataFrame(rows)
        records.append({
            'w2': w2,
            'MRR_mean': df['MRR_mean'].mean(),
            'MRR_std': df['MRR_mean'].std(),
            'Energy_mean': df['Energy_total'].mean(),
            'Energy_std': df['Energy_total'].std(),
            'Energy_per_pass': df['Energy_total'].sum() / df['passes'].sum(),
            'Time_mean': df['T_total'].mean(),
            'n_mean': df['n_mean'].mean(),
            'f_r_mean': df['f_r_mean'].mean(),
            'a_p_mean': df['a_p_mean'].mean(),
            'passes_mean': df['passes'].mean(),
            'Ra_viol_rate': df['Ra_violation'].mean(),
            'Success_rate': df['Success'].mean(),
        })
        print(f"  w2={w2:.1f}: MRR {records[-1]['MRR_mean']:.1f} mm3/min | "
              f"Energy {records[-1]['Energy_mean']:.1f} kJ | "
              f"Ra viol {records[-1]['Ra_viol_rate']*100:.1f}%")

    return pd.DataFrame(records)


def compare_with_sweep(conditioned_df, sweep_path=None):
    """Compare the conditioned frontier against the per-weight sweep."""
    if sweep_path is None:
        sweep_path = PROJECT_ROOT / "results" / "ablation" / "ablation_summary.csv"
    sweep_path = Path(sweep_path)
    if not sweep_path.exists():
        print(f"No sweep results at {sweep_path}, skipping the comparison")
        return None

    sweep = pd.read_csv(sweep_path)
    merged = conditioned_df.merge(sweep, on='w2', suffixes=('_conditioned', '_sweep'))

    merged['MRR_gap'] = merged['MRR_mean_conditioned'] - merged['MRR_mean_sweep']
    merged['MRR_gap_pct'] = 100.0 * merged['MRR_gap'] / merged['MRR_mean_sweep']
    merged['Energy_gap'] = merged['Energy_mean_conditioned'] - merged['Energy_mean_sweep']
    merged['Energy_gap_pct'] = 100.0 * merged['Energy_gap'] / merged['Energy_mean_sweep']

    return merged


def main():
    model_dir = PROJECT_ROOT / "models" / "PPO_conditioned_seed42"
    if not (model_dir / "final_model.zip").exists():
        raise FileNotFoundError(
            f"No conditioned policy at {model_dir}. Run agent/train_rl_conditioned.py first.")

    out_dir = PROJECT_ROOT / "results" / "conditioned"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Sweeping the energy weight for the conditioned policy...")
    frontier = evaluate_frontier(model_dir)
    frontier.to_csv(out_dir / "conditioned_frontier.csv", index=False)

    comparison = compare_with_sweep(frontier)
    if comparison is not None:
        comparison.to_csv(out_dir / "conditioned_vs_sweep.csv", index=False)
        print("\nConditioned policy against the per-weight sweep:")
        print(comparison[[
            'w2', 'MRR_mean_conditioned', 'MRR_mean_sweep', 'MRR_gap_pct',
            'Energy_mean_conditioned', 'Energy_mean_sweep', 'Energy_gap_pct'
        ]].round(2).to_string(index=False))

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
