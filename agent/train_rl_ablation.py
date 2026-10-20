"""
agent/train_rl_ablation.py

Energy Weight Ablation Study:
Trains PPO models across different energy weight balances w2 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
where w1 = 1.0 - w2.
Evaluates each policy across Monte Carlo scenarios to compute the empirical Pareto frontier (MRR vs Energy).
Outputs:
- results/ablation/ablation_results.csv
- results/plots/ablation_pareto_frontier.png
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from env.milling_env import MillingDTEnv
from agent.train_rl import train
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from viz.plotting_utils import set_style

def evaluate_ablation_model(model_dir, num_episodes=50, seed=42):
    model_path = model_dir / "final_model.zip"
    vec_path = model_dir / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    rows = []

    for ep in range(num_episodes):
        test_env = MillingDTEnv()
        vec_env = DummyVecEnv([lambda: test_env])
        vec_env = VecNormalize.load(str(vec_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        obs = vec_env.reset()
        done = False
        ep_MRR, ep_Energy, ep_Ra, ep_VB, ep_Time = [], [], [], [], []
        steps = 0

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = vec_env.step(act)
            info = info[0]
            steps += 1
            ep_MRR.append(info['MRR'])
            ep_Energy.append(info['Energy'])
            ep_Ra.append(info['Ra'])
            ep_VB.append(info['VB'])
            
            # Calculate Time (t_cut in seconds)
            L_cut = test_env.L_cut if hasattr(test_env, 'L_cut') else 100.0
            t_cut_s = (L_cut / (info['n'] * info['f_r'])) * 60.0
            ep_Time.append(t_cut_s)

            if done[0]:
                break

        rows.append({
            'ep_MRR_mean': np.mean(ep_MRR),
            'ep_Energy_total': sum(ep_Energy),
            'ep_Time_total': sum(ep_Time),
            'ep_Ra_max': max(ep_Ra),
            'ep_VB_max': max(ep_VB),
            'ra_violation': 1.0 if max(ep_Ra) > 1.6 else 0.0,
            'vb_violation': 1.0 if max(ep_VB) > 0.3 else 0.0,
            'passes': steps
        })

    df = pd.DataFrame(rows)
    return {
        'MRR_mean': df['ep_MRR_mean'].mean(),
        'MRR_std': df['ep_MRR_mean'].std(),
        'Energy_mean': df['ep_Energy_total'].mean(),
        'Energy_std': df['ep_Energy_total'].std(),
        'Time_mean': df['ep_Time_total'].mean(),
        'Time_std': df['ep_Time_total'].std(),
        'Ra_max_mean': df['ep_Ra_max'].mean(),
        'VB_max_mean': df['ep_VB_max'].mean(),
        'Ra_viol_rate': df['ra_violation'].mean(),
        'VB_viol_rate': df['vb_violation'].mean()
    }

def run_ablation(weights=[0.0, 0.5, 1.0], timesteps=100_000, seed=42):
    logs_dir = PROJECT_ROOT / "results" / "ablation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("=" * 65)
    print("  ENERGY WEIGHT ABLATION STUDY: PARETO FRONTIER EXPLORATION")
    print(f"  Weights w2 in {weights} | Steps per model: {timesteps:,}")
    print("=" * 65)

    for w2 in weights:
        w1 = round(1.0 - w2, 2)
        tag = f"w2_{w2:.1f}".replace('.', '_')
        save_dir = PROJECT_ROOT / "models" / f"ablation_{tag}"

        print(f"\n[+] Training Ablation Policy: w1={w1:.1f} (Throughput), w2={w2:.1f} (Energy)...")
        # Train with specific reward weights
        train(
            algo="PPO",
            timesteps=timesteps,
            seed=seed,
            device="cpu",
            progress_bar=False,
            save_dir=save_dir,
            env_kwargs={"w1": w1, "w2": w2}
        )

        print(f"    Evaluating policy in {save_dir}...")
        eval_metrics = evaluate_ablation_model(save_dir, num_episodes=50, seed=seed)
        record = {
            'w1': w1,
            'w2': w2,
            **eval_metrics
        }
        records.append(record)
        print(f"    -> MRR: {eval_metrics['MRR_mean']:.1f} mm/min | Energy: {eval_metrics['Energy_mean']:.1f} kJ | Ra Viol: {eval_metrics['Ra_viol_rate']*100:.1f}%")

    df_res = pd.DataFrame(records)
    csv_out = logs_dir / "ablation_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved ablation data to {csv_out}")

    # Plot Line Plots for Energy and Time
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Energy vs w2
    ax1.plot(df_res['w2'], df_res['Energy_mean'], '-o', color='#0d9488', linewidth=2.5, markersize=8, label='Total Energy')
    ax1.fill_between(df_res['w2'], 
                     df_res['Energy_mean'] - df_res['Energy_std'], 
                     df_res['Energy_mean'] + df_res['Energy_std'], 
                     color='#0d9488', alpha=0.2)
    ax1.set_title("Energy Consumption vs. Energy Weightage", fontweight='bold')
    ax1.set_xlabel("Energy Weight ($)")
    ax1.set_ylabel("Total Energy (kJ)")
    ax1.set_xticks([0.0, 0.5, 1.0])
    ax1.set_xticklabels(['0.0\n(Max Throughput)', '0.5\n(Balanced)', '1.0\n(Max Green)'])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Time vs w2
    ax2.plot(df_res['w2'], df_res['Time_mean'], '-o', color='#ea580c', linewidth=2.5, markersize=8, label='Cycle Time')
    ax2.fill_between(df_res['w2'], 
                     df_res['Time_mean'] - df_res['Time_std'], 
                     df_res['Time_mean'] + df_res['Time_std'], 
                     color='#ea580c', alpha=0.2)
    ax2.set_title("Machining Cycle Time vs. Energy Weightage", fontweight='bold')
    ax2.set_xlabel("Energy Weight ($)")
    ax2.set_ylabel("Total Cycle Time (s)")
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.set_xticklabels(['0.0\n(Max Throughput)', '0.5\n(Balanced)', '1.0\n(Max Green)'])
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_file = plots_dir / "ablation_line_plots.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\n[+] Saved Energy vs Time line plots to {plot_file}")
    return df_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--steps", type=int, default=100000, help="Timesteps per model")
    args = parser.parse_args()
    run_ablation(timesteps=args.steps)
