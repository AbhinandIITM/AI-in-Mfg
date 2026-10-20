"""
agent/evaluate.py

The one evaluation entry point of the project.

Two modes, both on the same policy:

    # the Monte Carlo sweep of the production policy at the default weight
    python agent/evaluate.py

    # the same policy asked for every point of the energy trade-off
    python agent/evaluate.py --sweep-weight

The weight is part of the observation, so the second mode is a query, not a
retrain.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import env # Triggers registration
from env import ENV_ID
from config.paths import MODELS_DIR, RESULTS_DIR, WEIGHT_SWEEP_DIR
from utils.seeding import set_seed
from config.physics_constants import ALL_CONSTANTS, ENGINEERING_CONSTRAINTS
from config.scenarios import generate_random_configs

# The energy weights the frontier is measured on.
WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

def evaluate_policy_on_scenarios(model, vec_norm_path, configs, seed=42, deterministic=True,
                                 env_id=ENV_ID, env_kwargs=None):
    """Run one policy over a set of scenarios, returning one row per scenario.

    The same scenario set is used for every policy that is compared, so the
    rows are directly comparable. The policy acts deterministically by
    default, so a sweep of the same policy over the same scenarios returns the
    same numbers every time.
    """
    rows = []
    extra_kwargs = dict(env_kwargs or {})

    for idx, (name, env_kwargs) in enumerate(configs.items()):
        eval_env = gym.make(env_id, **env_kwargs, **extra_kwargs)
        # Seed the underlying environment so every scenario is reproducible.
        # VecNormalize.reset() takes no seed, so the seed goes in first.
        eval_env.reset(seed=seed + idx)

        vec_env = DummyVecEnv([lambda: eval_env])
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        obs = vec_env.reset()
        done = False
        info = {}
        ep_MRR, ep_Energy, ep_Ra, ep_VB = [], [], [], []
        ep_n, ep_f, ep_ap = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info_list = vec_env.step(action)
            info = info_list[0]

            ep_MRR.append(info['MRR'])
            ep_Energy.append(info['Energy'])
            ep_Ra.append(info['Ra'])
            ep_VB.append(info['VB'])
            ep_n.append(info['n'])
            ep_f.append(info['f_r'])
            ep_ap.append(info['a_p'])

            done = bool(done[0])

        rows.append({
            'config': name,
            'MRR_mean': float(np.mean(ep_MRR)),
            'Energy_total': float(np.sum(ep_Energy)),
            'Ra_max': float(np.max(ep_Ra)),
            'Ra_final': float(ep_Ra[-1]),
            'VB_max': float(np.max(ep_VB)),
            'T_total': float(info.get('T_total', 0.0)),
            'n_mean': float(np.mean(ep_n)),
            'f_r_mean': float(np.mean(ep_f)),
            'a_p_mean': float(np.mean(ep_ap)),
            'passes': len(ep_MRR),
            'Ra_violation': int(np.max(ep_Ra) > ENGINEERING_CONSTRAINTS['Ra_max']),
            'Ra_final_violation': int(ep_Ra[-1] > ENGINEERING_CONSTRAINTS['Ra_max']),
            'Ra_safe_violation': int(np.max(ep_Ra) > ENGINEERING_CONSTRAINTS['Ra_max']
                                     * ENGINEERING_CONSTRAINTS.get('safety_factor', 0.85)),
            'Ra_final_safe_violation': int(ep_Ra[-1] > ENGINEERING_CONSTRAINTS['Ra_max']
                                           * ENGINEERING_CONSTRAINTS.get('safety_factor', 0.85)),
            'VB_violation': int(np.max(ep_VB) > ENGINEERING_CONSTRAINTS['VB_max']),
            'Broken': int(bool(info.get('broken', False))),
            'Success': int(bool(info.get('is_finishing', False))
                           and not info.get('broken', False)
                           and np.max(ep_VB) <= ENGINEERING_CONSTRAINTS['VB_max'])
        })

        vec_env.close()
        eval_env.close()

    return rows

def compute_statistics(df, ci=0.95):
    """
    Compute mean, std, and confidence intervals for metrics.
    """
    stats_dict = {}
    metrics = ['MRR_mean', 'Energy_total', 'Ra_max', 'VB_max', 'T_total']

    for col in metrics:
        if col not in df.columns: continue
        data = df[col]
        mean = data.mean()
        std = data.std()
        n = len(data)

        # t-distribution based CI
        ci_h = std / np.sqrt(n) * stats.t.ppf((1 + ci) / 2, n - 1)

        stats_dict[col] = {
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci_h,
            'ci_upper': mean + ci_h
        }

    # Violation rates (Binomial CI)
    for col in ['Ra_violation', 'Ra_final_violation', 'Ra_safe_violation',
                'Ra_final_safe_violation', 'VB_violation', 'Broken']:
        if col not in df.columns: continue
        rate = df[col].mean()
        n = len(df)

        # Normal approximation for binomial CI
        z = stats.norm.ppf((1 + ci) / 2)
        ci_h = z * np.sqrt(rate * (1 - rate) / n)

        stats_dict[col] = {
            'rate': rate,
            'ci_lower': max(0, rate - ci_h),
            'ci_upper': min(1, rate + ci_h)
        }

    return stats_dict

def find_model_dir(algo="PPO"):
    """Locate a trained policy directory, preferring the seed 42 run."""
    candidates = [
        MODELS_DIR / f"{algo}_seed42",
        MODELS_DIR / f"{algo}_seed1",
        MODELS_DIR / algo
    ] + list(MODELS_DIR.glob(f"{algo}_seed*"))

    for cand in candidates:
        if (cand / "final_model.zip").exists():
            return cand
    return None

def evaluate(num_scenarios=100, seed=42, algo="PPO", w2=None, model_dir=None):
    """The Monte Carlo sweep of the policy at one energy weight."""
    set_seed(seed)
    logs_dir = RESULTS_DIR / "monte_carlo"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  MONTE CARLO INFERENCE: STATISTICAL RIGOR SWEEP ({algo})")
    print("=" * 70)

    model_dir = Path(model_dir) if model_dir else find_model_dir(algo)
    if model_dir is None or not (model_dir / "final_model.zip").exists():
        print(f"Skipping {algo}: Model not found in models/")
        return None

    model_path = model_dir / "final_model.zip"
    vec_norm_path = model_dir / "vecnormalize.pkl"

    print(f"Loading {algo} model from {model_dir}...")
    model = PPO.load(str(model_path), device="cpu")

    configs = generate_random_configs(num_scenarios, seed=seed)
    env_kwargs = {"w2": w2} if w2 is not None else {}

    print(f"Running sweep for {algo}...")
    scenario_metrics = evaluate_policy_on_scenarios(
        model, vec_norm_path, configs, seed=seed, env_kwargs=env_kwargs
    )

    df_algo = pd.DataFrame(scenario_metrics)
    df_algo.to_csv(logs_dir / f"eval_{algo}_MC_v5.csv", index=False)

    stats_dict = compute_statistics(df_algo)

    print(f"\nResults for {algo} over {len(df_algo)} scenarios:")
    print(f"  MRR:        {stats_dict['MRR_mean']['mean']:.2f} +/- {stats_dict['MRR_mean']['std']:.2f} [95% CI: {stats_dict['MRR_mean']['ci_lower']:.2f}-{stats_dict['MRR_mean']['ci_upper']:.2f}]")
    print(f"  Energy:     {stats_dict['Energy_total']['mean']:.2f} +/- {stats_dict['Energy_total']['std']:.2f} [95% CI: {stats_dict['Energy_total']['ci_lower']:.2f}-{stats_dict['Energy_total']['ci_upper']:.2f}]")
    print(f"  Cycle time: {stats_dict['T_total']['mean']:.2f} +/- {stats_dict['T_total']['std']:.2f} s [95% CI: {stats_dict['T_total']['ci_lower']:.2f}-{stats_dict['T_total']['ci_upper']:.2f}]")
    print(f"  Ra Viol:    {stats_dict['Ra_violation']['rate']*100:.1f}% [95% CI: {stats_dict['Ra_violation']['ci_lower']*100:.1f}-{stats_dict['Ra_violation']['ci_upper']*100:.1f}%] (worst pass)")
    print(f"  Ra Final:   {stats_dict['Ra_final_violation']['rate']*100:.1f}% [95% CI: {stats_dict['Ra_final_violation']['ci_lower']*100:.1f}-{stats_dict['Ra_final_violation']['ci_upper']*100:.1f}%] (finished surface)")
    print(f"  VB Viol:    {stats_dict['VB_violation']['rate']*100:.1f}%")
    print(f"  Broken:     {stats_dict['Broken']['rate']*100:.1f}%")
    print(f"  Success:    {df_algo['Success'].mean()*100:.1f}%")
    print(f"  Mean n:     {df_algo['n_mean'].mean():.0f} RPM | Mean f_r: {df_algo['f_r_mean'].mean():.3f} mm/r")

    summary = [{
        'Algorithm': algo,
        'w2': w2 if w2 is not None else float(ALL_CONSTANTS['reward']['w2']),
        'MRR_mean': stats_dict['MRR_mean']['mean'],
        'MRR_ci_lower': stats_dict['MRR_mean']['ci_lower'],
        'MRR_ci_upper': stats_dict['MRR_mean']['ci_upper'],
        'Energy_mean': stats_dict['Energy_total']['mean'],
        'Energy_ci_lower': stats_dict['Energy_total']['ci_lower'],
        'Energy_ci_upper': stats_dict['Energy_total']['ci_upper'],
        'T_total_mean': stats_dict['T_total']['mean'],
        'Ra_violation_rate': stats_dict['Ra_violation']['rate'],
        'Ra_final_violation_rate': stats_dict['Ra_final_violation']['rate'],
        'VB_violation_rate': stats_dict['VB_violation']['rate'],
        'Broken_rate': stats_dict['Broken']['rate'],
        'Success_rate': df_algo['Success'].mean(),
    }]
    pd.DataFrame(summary).to_csv(logs_dir / "eval_summary_v5.csv", index=False)

    return df_algo

def evaluate_weight_sweep(weights=WEIGHTS, num_scenarios=50, seed=42, model_dir=None):
    """Ask the same policy for every point of the energy trade-off.

    The weight is part of the observation, so this is a query of one policy,
    not a retrain per weight.
    """
    model_dir = Path(model_dir) if model_dir else find_model_dir("PPO")
    if model_dir is None or not (model_dir / "final_model.zip").exists():
        raise FileNotFoundError("No trained policy found. Run agent/train_rl.py first.")

    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    vec_norm_path = model_dir / "vecnormalize.pkl"

    configs = generate_random_configs(num_scenarios, seed=seed)

    records = []
    for w2 in weights:
        set_seed(seed)
        rows = evaluate_policy_on_scenarios(
            model, vec_norm_path, configs, seed=seed,
            env_kwargs={"w2": w2},
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
            'Ra_viol_rate': df['Ra_violation'].mean(),
            'Success_rate': df['Success'].mean(),
        })
        print(f"  w2={w2:.1f}: MRR {records[-1]['MRR_mean']:.1f} mm3/min | "
              f"Energy {records[-1]['Energy_mean']:.1f} kJ | "
              f"n {records[-1]['n_mean']:.0f} RPM | "
              f"Ra viol {records[-1]['Ra_viol_rate']*100:.1f}%")

    return pd.DataFrame(records)

def compare_with_ablation(frontier, ablation_path=None):
    """Compare the runtime weight sweep against the per-weight policies.

    The sweep is one policy asked for every point of the trade-off. The
    ablation is one policy per weight, each trained without ever seeing the
    weight. The comparison is the claim that a single policy covers the same
    frontier as a policy per point.
    """
    if ablation_path is None:
        ablation_path = RESULTS_DIR / "ablation" / "ablation_summary.csv"
    ablation_path = Path(ablation_path)
    if not ablation_path.exists():
        print(f"No ablation results at {ablation_path}, skipping the comparison")
        return None

    ablation = pd.read_csv(ablation_path)
    merged = frontier.merge(ablation, on='w2', suffixes=('_sweep', '_ablation'))

    merged['MRR_gap_pct'] = 100.0 * (merged['MRR_mean_sweep'] - merged['MRR_mean_ablation']) / merged['MRR_mean_ablation']
    merged['Energy_gap_pct'] = 100.0 * (merged['Energy_mean_sweep'] - merged['Energy_mean_ablation']) / merged['Energy_mean_ablation']

    return merged

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the trained policy")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w2", type=float, default=None,
                        help="the energy weight to evaluate at")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--sweep-weight", action="store_true",
                        help="sweep the energy weight at runtime and write the frontier")
    parser.add_argument("--compare", action="store_true",
                        help="compare the weight sweep against the per-weight policies")
    args = parser.parse_args()

    if args.sweep_weight:
        out_dir = WEIGHT_SWEEP_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        frontier = evaluate_weight_sweep(num_scenarios=args.scenarios, seed=args.seed,
                                        model_dir=args.model_dir)
        frontier.to_csv(out_dir / "weight_sweep.csv", index=False)
        print(f"\nSaved to {out_dir / 'weight_sweep.csv'}")

        if args.compare:
            comparison = compare_with_ablation(frontier)
            if comparison is not None:
                comparison.to_csv(out_dir / "weight_sweep_vs_ablation.csv", index=False)
                print("\nOne policy against the per-weight policies:")
                print(comparison[[
                    'w2', 'MRR_mean_sweep', 'MRR_mean_ablation', 'MRR_gap_pct',
                    'Energy_mean_sweep', 'Energy_mean_ablation', 'Energy_gap_pct'
                ]].round(2).to_string(index=False))
    else:
        evaluate(num_scenarios=args.scenarios, seed=args.seed, w2=args.w2,
                 model_dir=args.model_dir)

if __name__ == "__main__":
    main()
