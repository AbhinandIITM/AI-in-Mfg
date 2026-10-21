"""
viz/plot_baselines_comparison.py

Generates publication-quality comparison charts:
1. PPO vs Non-RL Baselines (Constant High, Constant Safe, Rule-Based, NSGA-II)
   across MRR, Total Energy, and Cumulative Reward.
2. Exports to experiments/plots/baseline_vs_ppo_comparison.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from viz.plotting_utils import set_style
from MillingDT.config import ALL_CONSTANTS, EVALUATION_DIR, PLOTS_DIR, PRIMARY_MODEL_DIR, PRIMARY_RUN
import MillingDT.env
from MillingDT.env.milling_env import MillingDTEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# The learned policy is queried at the configured energy weight, the same
# weight the classical baselines are scored at. It is pinned here so the
# table states the weight it was measured at.
QUERY_WEIGHT = float(ALL_CONSTANTS['reward']['w2'])

def collect_ppo_baseline_eval(n_episodes=50, seed=42):
    model_dir = PRIMARY_MODEL_DIR
    if not (model_dir / "final_model.zip").exists():
        raise FileNotFoundError(
            f"PPO model not found at experiments/run_{PRIMARY_RUN}/models. "
            "Train the primary run first.")

    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    rows = []

    for ep in range(n_episodes):
        test_env = MillingDTEnv(w2=QUERY_WEIGHT)
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
            "energy_weight": QUERY_WEIGHT,
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
    baselines_path = EVALUATION_DIR / "baselines.csv"
    if not baselines_path.exists():
        print("Running run_baselines.py first...")
        from experiments.run_baselines import evaluate_all_baselines
        evaluate_all_baselines()

    df_base = pd.read_csv(baselines_path)
    df_ppo = collect_ppo_baseline_eval(n_episodes=50)

    # Combine
    df_all = pd.concat([df_base, df_ppo], ignore_index=True)

    # Map policy labels for clear presentation. Every baseline that carries a
    # row in the table needs a label here, otherwise the bar is dropped from
    # the figure without a word.
    label_map = {
        "constant_high": "Constant\nHigh",
        "constant_safe": "Constant\nSafe",
        "rule_based": "Rule-\nBased",
        "nsga2": "NSGA-II\nStatic",
        "receding_horizon": "Receding-\nHorizon",
        "PPO (Ours)": "Dynamic PPO\n(ours)"
    }
    df_all["policy_label"] = df_all["policy"].map(label_map)
    missing = sorted(set(df_all["policy"]) - set(label_map))
    if missing:
        raise KeyError(f"Baselines without a label in the figure: {missing}")

    # Order, from the weakest policy to the learned one.
    order = [
        "Constant\nSafe",
        "Rule-\nBased",
        "NSGA-II\nStatic",
        "Constant\nHigh",
        "Receding-\nHorizon",
        "Dynamic PPO\n(ours)"
    ]
    palette = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#64748b", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    # 1. MRR (Throughput)
    sns.barplot(
        data=df_all, x="policy_label", y="mean_mrr_mm3_min",
        order=order, palette=palette, ax=axes[0], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[0].set_title("Machining Throughput (MRR)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean MRR (mm³/min)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", labelsize=9)

    # 2. Total Energy (Green Manufacturing)
    # Use log scale or direct scale note if constant_safe is high
    energy_order = [o for o in order if "Safe" not in o]
    energy_palette = [c for o, c in zip(order, palette) if "Safe" not in o]
    sns.barplot(
        data=df_all[df_all["policy"] != "constant_safe"], x="policy_label", y="total_energy_kJ",
        order=energy_order, palette=energy_palette, ax=axes[1], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[1].set_title("Total Energy Consumption (kJ)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Energy per Part (kJ) [Lower is Better]", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", labelsize=9)

    # 3. Overall Multi-Objective Objective / Reward
    sns.barplot(
        data=df_all, x="policy_label", y="total_reward",
        order=order, palette=palette, ax=axes[2], capsize=0.1, err_kws={'linewidth': 1.5}
    )
    axes[2].set_title("Multi-Objective Reward Performance", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Total Cumulative Reward [Higher is Better]", fontsize=11)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].tick_params(axis="x", labelsize=9)

    # Highlight PPO in reward plot
    axes[2].axhline(df_ppo["total_reward"].mean(), color="#2ca02c", linestyle=":", alpha=0.7, label=f"PPO Mean: {df_ppo['total_reward'].mean():.1f}")
    axes[2].legend(loc="lower right")

    plt.suptitle("Benchmark: Dynamic DRL (PPO) vs. Classical & Metaheuristic Baselines\n"
                 f"The learned policy is queried at the energy weight w2 = {QUERY_WEIGHT:.2f}",
                 fontsize=15, fontweight="bold", y=1.06)
    # The context a reader needs: the four static policies are infeasible, so
    # the comparison that matters is the receding-horizon controller.
    fig.text(0.5, -0.02,
             "The four static policies finish outside the roughness safety limit, "
             "which is what their negative reward measures.\n"
             "The receding-horizon controller and the learned policy finish every "
             "part in spec; the controller removes more material per minute.",
             ha="center", va="top", fontsize=11, color="#334155")
    plt.tight_layout()

    out_plot = PLOTS_DIR / "baseline_vs_ppo_comparison.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"\nThe learned policy was queried at the energy weight w2 = {QUERY_WEIGHT:.2f}. "
          "The classical baselines are scored at the same weight.")

if __name__ == "__main__":
    plot_comparison()
