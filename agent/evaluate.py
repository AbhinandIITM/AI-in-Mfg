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
from stable_baselines3.common.vec_env import VecNormalize


import env # Triggers registration
from utils.seeding import set_seed
from config.physics_constants import ENGINEERING_CONSTRAINTS

def generate_random_configs(n=100, seed=42):
    """
    Generate randomized machining scenarios for Monte Carlo sweep.
    """
    np.random.seed(seed)
    configs = {}
    for i in range(n):
        configs[f"Random_{i:03d}"] = {
            "kappa_total": np.random.uniform(2.0, 15.0),
            "D": np.random.uniform(40.0, 150.0),
            "L_cut": np.random.uniform(50.0, 200.0),
            "C_wear": np.random.uniform(1.0e-2, 6.0e-2),
            "k1": np.random.uniform(10.0, 15.0),
            "k2": np.random.uniform(0.5, 1.2),
        }
    return configs

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
    for col in ['Ra_violation', 'VB_violation', 'Broken']:
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

def evaluate(num_scenarios=100, seed=42):
    set_seed(seed)
    logs_dir = PROJECT_ROOT / "results" / "monte_carlo"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MASSIVE MONTE CARLO INFERENCE: STATISTICAL RIGOR SWEEP (PPO)")
    print("=" * 70)

    algos = ["PPO"]
    configs = generate_random_configs(num_scenarios, seed=seed)

    summary_results = []

    for algo in algos:
        # Search for available PPO models (prefer seed42, seed1, then any PPO_seed*)
        candidate_dirs = [
            PROJECT_ROOT / "models" / f"{algo}_seed42",
            PROJECT_ROOT / "models" / f"{algo}_seed1",
            PROJECT_ROOT / "models" / algo
        ] + list((PROJECT_ROOT / "models").glob(f"{algo}_seed*"))

        model_dir = None
        for cand in candidate_dirs:
            if (cand / "final_model.zip").exists():
                model_dir = cand
                break

        if model_dir is None:
            print(f"Skipping {algo}: Model not found in models/")
            continue

        model_path = model_dir / "final_model.zip"
        vec_norm_path = model_dir / "vecnormalize.pkl"

        print(f"Loading {algo} model from {model_dir}...")
        model = PPO.load(str(model_path), device="cpu")



        scenario_metrics = []

        print(f"Running sweep for {algo}...")
        for conf_name, env_kwargs in tqdm(configs.items()):
            # Create a specialized env for this scenario
            eval_env = gym.make("MillingDTEnv-v1", **env_kwargs)

            # Load normalization stats
            # VecNormalize requires a vectorized env
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: eval_env])
            vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

            obs = vec_env.reset()
            done = False

            ep_MRR = []
            ep_Energy = []
            ep_Ra = []
            ep_VB = []
            ep_T = 0

            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = vec_env.step(action)
                info = info[0]

                ep_MRR.append(info['MRR'])
                ep_Energy.append(info.get('Energy', 0))
                ep_Ra.append(info['Ra'])
                ep_VB.append(info['VB'])
                ep_T = info.get('T_total', 0) # Accumulated time
                done = info.get('terminated', done) or info.get('truncated', False)

            scenario_metrics.append({
                'config': conf_name,
                'MRR_mean': np.mean(ep_MRR),
                'Energy_total': np.sum(ep_Energy),
                'Ra_max': np.max(ep_Ra),
                'VB_max': np.max(ep_VB),
                'T_total': ep_T,
                'Ra_violation': 1 if np.max(ep_Ra) > ENGINEERING_CONSTRAINTS['Ra_max'] else 0,
                'VB_violation': 1 if np.max(ep_VB) > ENGINEERING_CONSTRAINTS['VB_max'] else 0,
                'Broken': 1 if info.get('broken', False) else 0,
                'Success': 1 if info.get('is_finishing', False) and not info.get('broken', False) and info['VB'] <= ENGINEERING_CONSTRAINTS['VB_max'] else 0
            })

        df_algo = pd.DataFrame(scenario_metrics)
        df_algo.to_csv(logs_dir / f"eval_{algo}_MC_v4.csv", index=False)

        stats_dict = compute_statistics(df_algo)

        print(f"\nResults for {algo}:")
        print(f"  MRR:        {stats_dict['MRR_mean']['mean']:.2f} ± {stats_dict['MRR_mean']['std']:.2f} [95% CI: {stats_dict['MRR_mean']['ci_lower']:.2f}-{stats_dict['MRR_mean']['ci_upper']:.2f}]")
        print(f"  Energy:     {stats_dict['Energy_total']['mean']:.2f} ± {stats_dict['Energy_total']['std']:.2f} [95% CI: {stats_dict['Energy_total']['ci_lower']:.2f}-{stats_dict['Energy_total']['ci_upper']:.2f}]")
        print(f"  Ra Viol:    {stats_dict['Ra_violation']['rate']*100:.1f}% [95% CI: {stats_dict['Ra_violation']['ci_lower']*100:.1f}-{stats_dict['Ra_violation']['ci_upper']*100:.1f}%]")
        print(f"  VB Viol:    {stats_dict['VB_violation']['rate']*100:.1f}%")
        print(f"  Success:    {df_algo['Success'].mean()*100:.1f}%")

        summary_results.append({
            'Algorithm': algo,
            'MRR_mean': stats_dict['MRR_mean']['mean'],
            'Energy_mean': stats_dict['Energy_total']['mean'],
            'Success_rate': df_algo['Success'].mean()
        })

    pd.DataFrame(summary_results).to_csv(logs_dir / "eval_summary_v4.csv", index=False)

if __name__ == "__main__":
    evaluate()
