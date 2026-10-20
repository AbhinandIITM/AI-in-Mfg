"""
viz/plot_baselines_comparison.py

Generates publication-quality comparison charts:
1. PPO vs Non-RL Baselines (Constant High, Constant Safe, Rule-Based, NSGA-II)
   across MRR, Total Energy, and Cumulative Reward.
2. Exports to results/plots/baseline_vs_ppo_comparison.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from viz.plotting_utils import set_style
import env
from env.milling_env import MillingDTEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def collect_ppo_baseline_eval(n_episodes=50, seed=42):
    model_dir = PROJECT_ROOT / "models" / "PPO_seed42"
    if not (model_dir / "final_model.zip").exists():
        raise FileNotFoundError("PPO model not found at models/PPO_seed42")

    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    rows = []

    for ep in range(n_episodes):
        test_env = MillingDTEnv()
        vec_env = DummyVecEnv([lambda: test_env])
        vec_env = VecNormalize.load(str(model_dir / "vecnormalize.pkl"), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        obs = vec_env.reset()
        done = False
        ep_MRR, ep_Energy, ep_Ra, ep_VB, ep_n, ep_f = [], [], [], [], [], []
        total_rew = 0.0
        steps = 0

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = vec_env.step(act)
            info = info[0]
            steps += 1
            total_rew += rew[0]
            ep_MRR.append(info['MRR'])
            ep_Energy.append(info['Energy'])
            ep_Ra.append(info['Ra'])
            ep_VB.append(info['VB'])
            ep_n.append(info['n'])
            ep_f.append(info['f_r'])
            if done[0]:
                break

        rows.append({
            "policy": "PPO (Ours)",
            "episode": ep,
            "total_reward": total_rew,
            "passes": steps,
            "broken": info.get("broken", False),
            "ra_violation_rate": 1.0 if max(ep_Ra) > 1.6 else 0.0,
            "total_energy_kJ": sum(ep_Energy),
            "mean_mrr_mm3_min": np.mean(ep_MRR),
            "mean_n_rpm": np.mean(ep_n),
            "mean_f_r_mm_rev": np.mean(ep_f),
            "cycle_time_s": info.get("T_total", 0.0),
        })

    return pd.DataFrame(rows)

def plot_comparison():
    set_style()
    baselines_path = PROJECT_ROOT / "results" / "baseline_results_v5.csv"
    if not baselines_path.exists():
        print("Running run_baselines.py first...")
        from baselines.run_baselines import evaluate_all_baselines
        evaluate_all_baselines()

    df_base = pd.read_csv(baselines_path)
    df_ppo = collect_ppo_baseline_eval(n_episodes=50)

    # Combine
    df_all = pd.concat([df_base, df_ppo], ignore_index=True)

    # Map policy labels for clear presentation
    label_map = {
        "constant_high": "Constant High\n(Max Speed/Feed)",
        "constant_safe": "Constant Safe\n(Min Speed/Feed)",
        "rule_based": "Rule-Based\n(Feed/Speed Decay)",
        "nsga2": "NSGA-II Static\n(Pareto Knee)",
        "PPO (Ours)": "Dynamic PPO\n(Deep RL - Ours)"
    }
    df_all["policy_label"] = df_all["policy"].map(label_map)

    # Order
    order = [
        "Constant Safe\n(Min Speed/Feed)",
        "Rule-Based\n(Feed/Speed Decay)",
        "NSGA-II Static\n(Pareto Knee)",
        "Constant High\n(Max Speed/Feed)",
        "Dynamic PPO\n(Deep RL - Ours)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    palette = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#2ca02c"]

    # 1. MRR (Throughput)
    sns.barplot(
        data=df_all, x="policy_label", y="mean_mrr_mm3_min",
        order=order, palette=palette, ax=axes[0], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[0].set_title("Machining Throughput (MRR)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean MRR (mm³/min)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Total Energy (Green Manufacturing)
    # Use log scale or direct scale note if constant_safe is high
    sns.barplot(
        data=df_all[df_all["policy"] != "constant_safe"], x="policy_label", y="total_energy_kJ",
        order=[o for o in order if "Safe" not in o], palette=palette[1:], ax=axes[1], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[1].set_title("Total Energy Consumption (kJ)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Energy per Part (kJ) [Lower is Better]", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Overall Multi-Objective Objective / Reward
    sns.barplot(
        data=df_all, x="policy_label", y="total_reward",
        order=order, palette=palette, ax=axes[2], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[2].set_title("Multi-Objective Reward Performance", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Total Cumulative Reward [Higher is Better]", fontsize=11)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight PPO in reward plot
    axes[2].axhline(df_ppo["total_reward"].mean(), color="#2ca02c", linestyle=":", alpha=0.7, label=f"PPO Mean: {df_ppo['total_reward'].mean():.1f}")
    axes[2].legend(loc="lower right")

    plt.suptitle("Benchmark: Dynamic DRL (PPO) vs. Classical & Metaheuristic Baselines", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_plot = PROJECT_ROOT / "results" / "plots" / "baseline_vs_ppo_comparison.png"
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {out_plot}")

    # Print summary statistics table
    summary = df_all.groupby("policy").agg({
        "mean_mrr_mm3_min": ["mean", "std"],
        "total_energy_kJ": ["mean", "std"],
        "total_reward": ["mean", "std"],
        "ra_violation_rate": "mean",
        "broken": "mean"
    })
    print("\nBenchmark Summary Table:")
    print(summary)

if __name__ == "__main__":
    plot_comparison()
