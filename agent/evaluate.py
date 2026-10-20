"""
agent/evaluate.py

Monte Carlo robustness sweep for a trained policy.

Every scenario perturbs the calibrated operating point: workpiece geometry,
material and tool coefficients, and the initial flank wear of the tool. The
sweep reports means with 95% confidence intervals so that a robustness claim
carries its own uncertainty.
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
from utils.seeding import set_seed
from config.physics_constants import ENGINEERING_CONSTRAINTS
from config.scenarios import generate_random_configs

from config.scenarios import generate_random_configs as _generate_random_configs

def generate_random_configs(n=100, seed=42):
    """Generate randomized machining scenarios for the Monte Carlo sweep.

    Kept as a module level name for callers that import it from here. The
    implementation lives in config/scenarios.py, which is also what the
    environment samples its training scenarios from.
    """
    return _generate_random_configs(n, seed)

def evaluate_policy_on_scenarios(model, vec_norm_path, configs, seed=42, deterministic=False,
                                 env_id="MillingDTEnv-v1", env_kwargs=None):
    """Run one policy over a set of scenarios, returning one row per scenario.

    The same scenario set is used for every policy that is compared, so the
    rows are directly comparable.
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
            'VB_max': float(np.max(ep_VB)),
            'T_total': float(info.get('T_total', 0.0)),
            'n_mean': float(np.mean(ep_n)),
            'f_r_mean': float(np.mean(ep_f)),
            'a_p_mean': float(np.mean(ep_ap)),
            'passes': len(ep_MRR),
            'Ra_violation': int(np.max(ep_Ra) > ENGINEERING_CONSTRAINTS['Ra_max']),
            'Ra_safe_violation': int(np.max(ep_Ra) > ENGINEERING_CONSTRAINTS['Ra_max']
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
    for col in ['Ra_violation', 'Ra_safe_violation', 'VB_violation', 'Broken']:
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
        PROJECT_ROOT / "models" / f"{algo}_seed42",
        PROJECT_ROOT / "models" / f"{algo}_seed1",
        PROJECT_ROOT / "models" / algo
    ] + list((PROJECT_ROOT / "models").glob(f"{algo}_seed*"))

    for cand in candidates:
        if (cand / "final_model.zip").exists():
            return cand
    return None

def evaluate(num_scenarios=100, seed=42, algo="PPO"):
    set_seed(seed)
    logs_dir = PROJECT_ROOT / "results" / "monte_carlo"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  MONTE CARLO INFERENCE: STATISTICAL RIGOR SWEEP ({algo})")
    print("=" * 70)

    model_dir = find_model_dir(algo)
    if model_dir is None:
        print(f"Skipping {algo}: Model not found in models/")
        return None

    model_path = model_dir / "final_model.zip"
    vec_norm_path = model_dir / "vecnormalize.pkl"

    print(f"Loading {algo} model from {model_dir}...")
    model = PPO.load(str(model_path), device="cpu")

    configs = generate_random_configs(num_scenarios, seed=seed)

    print(f"Running sweep for {algo}...")
    scenario_metrics = evaluate_policy_on_scenarios(
        model, vec_norm_path, configs, seed=seed
    )

    df_algo = pd.DataFrame(scenario_metrics)
    df_algo.to_csv(logs_dir / f"eval_{algo}_MC_v5.csv", index=False)

    stats_dict = compute_statistics(df_algo)

    print(f"\nResults for {algo} over {len(df_algo)} scenarios:")
    print(f"  MRR:        {stats_dict['MRR_mean']['mean']:.2f} +/- {stats_dict['MRR_mean']['std']:.2f} [95% CI: {stats_dict['MRR_mean']['ci_lower']:.2f}-{stats_dict['MRR_mean']['ci_upper']:.2f}]")
    print(f"  Energy:     {stats_dict['Energy_total']['mean']:.2f} +/- {stats_dict['Energy_total']['std']:.2f} [95% CI: {stats_dict['Energy_total']['ci_lower']:.2f}-{stats_dict['Energy_total']['ci_upper']:.2f}]")
    print(f"  Cycle time: {stats_dict['T_total']['mean']:.2f} +/- {stats_dict['T_total']['std']:.2f} s [95% CI: {stats_dict['T_total']['ci_lower']:.2f}-{stats_dict['T_total']['ci_upper']:.2f}]")
    print(f"  Ra Viol:    {stats_dict['Ra_violation']['rate']*100:.1f}% [95% CI: {stats_dict['Ra_violation']['ci_lower']*100:.1f}-{stats_dict['Ra_violation']['ci_upper']*100:.1f}%]")
    print(f"  Ra Safe:    {stats_dict['Ra_safe_violation']['rate']*100:.1f}% (above the 0.85 safety limit)")
    print(f"  VB Viol:    {stats_dict['VB_violation']['rate']*100:.1f}% [95% CI: {stats_dict['VB_violation']['ci_lower']*100:.1f}-{stats_dict['VB_violation']['ci_upper']*100:.1f}%]")
    print(f"  Broken:     {stats_dict['Broken']['rate']*100:.1f}%")
    print(f"  Success:    {df_algo['Success'].mean()*100:.1f}%")
    print(f"  Mean n:     {df_algo['n_mean'].mean():.0f} RPM | Mean f_r: {df_algo['f_r_mean'].mean():.3f} mm/r")

    summary = [{
        'Algorithm': algo,
        'MRR_mean': stats_dict['MRR_mean']['mean'],
        'MRR_ci_lower': stats_dict['MRR_mean']['ci_lower'],
        'MRR_ci_upper': stats_dict['MRR_mean']['ci_upper'],
        'Energy_mean': stats_dict['Energy_total']['mean'],
        'Energy_ci_lower': stats_dict['Energy_total']['ci_lower'],
        'Energy_ci_upper': stats_dict['Energy_total']['ci_upper'],
        'T_total_mean': stats_dict['T_total']['mean'],
        'Ra_violation_rate': stats_dict['Ra_violation']['rate'],
        'VB_violation_rate': stats_dict['VB_violation']['rate'],
        'Broken_rate': stats_dict['Broken']['rate'],
        'Success_rate': df_algo['Success'].mean(),
    }]
    pd.DataFrame(summary).to_csv(logs_dir / "eval_summary_v5.csv", index=False)

    return df_algo

if __name__ == "__main__":
    evaluate()
