"""
experiments/controller_experiment.py

Does starting from the classical controller beat the controller?

Three arms, all evaluated on the same 100 randomized scenarios at the same
energy weight:

1. the receding-horizon controller itself,
2. the production policy, trained from scratch with PPO,
3. a policy whose actor is initialized from the controller and then
   fine-tuned with PPO.

The question is not whether the initialization helps the learning, it is
whether the learned policy ends up better than the expert it started from.

Outputs:
    results/evaluation/controller_init_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import MillingDT.env
from stable_baselines3 import PPO

from MillingDT.agent.controller_init import controller_action
from experiments.evaluate import evaluate_policy_on_scenarios
from MillingDT.config import (
    CONTROLLER_INITIALIZED_RUN,
    EVALUATION_DIR,
    PRIMARY_RUN,
    run_models,
)
from MillingDT.config.scenarios import generate_random_configs
from MillingDT.utils.seeding import set_seed

# The weight every arm is asked for.
WEIGHT = 0.4


def evaluate_controller(scenarios, seed=42, weight=WEIGHT):
    """Roll the controller out on the same scenarios, with the same metrics."""
    rows = []
    for idx, (name, scenario) in enumerate(scenarios.items()):
        eval_env = gym.make("MillingDTEnv-v1", w2=weight, **scenario)
        eval_env.reset(seed=seed + idx)
        unwrapped = eval_env.unwrapped

        obs, _ = eval_env.reset(seed=seed + idx)
        done = False
        ep_MRR, ep_Energy, ep_Ra, ep_VB = [], [], [], []
        ep_n, ep_f, ep_ap = [], [], []
        info = {}
        total_reward = 0.0
        while not done:
            action = controller_action(unwrapped)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            ep_MRR.append(info['MRR'])
            ep_Energy.append(info['Energy'])
            ep_Ra.append(info['Ra'])
            ep_VB.append(info['VB'])
            ep_n.append(info['n'])
            ep_f.append(info['f_r'])
            ep_ap.append(info['a_p'])

        rows.append({
            'config': name,
            'reward': total_reward,
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
            'Ra_violation': int(np.max(ep_Ra) > 1.6),
            'Ra_final_violation': int(ep_Ra[-1] > 1.6),
            'VB_violation': int(np.max(ep_VB) > 0.3),
            'Broken': int(bool(info.get('broken', False))),
            'Success': int(bool(info.get('is_finishing', False))
                           and not info.get('broken', False)),
        })
        eval_env.close()

    return rows


def summarize(name, rows):
    df = pd.DataFrame(rows)
    return {
        'arm': name,
        'mean_reward': df['reward'].mean(),
        'reward_std': df['reward'].std(),
        'MRR_mean': df['MRR_mean'].mean(),
        'MRR_std': df['MRR_mean'].std(),
        'Energy_per_part': df['Energy_total'].mean(),
        'Cycle_time': df['T_total'].mean(),
        'Ra_violation_worst_pass': df['Ra_violation'].mean(),
        'Ra_violation_finished_surface': df['Ra_final_violation'].mean(),
        'VB_violation': df['VB_violation'].mean(),
        'Broken': df['Broken'].mean(),
        'Success': df['Success'].mean(),
    }


def main():
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    controller_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    out_dir = EVALUATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    scenarios = generate_random_configs(100, seed=42)

    # 1. The controller.
    print("[1/3] Evaluating the receding-horizon controller")
    controller_rows = evaluate_controller(scenarios)
    controller = summarize("Receding-horizon controller", controller_rows)
    print(f"      MRR {controller['MRR_mean']:.1f} | energy {controller['Energy_per_part']:.1f} kJ | "
          f"Ra viol (finished) {controller['Ra_violation_finished_surface']*100:.1f}% | "
          f"success {controller['Success']*100:.1f}%")

    # 2. The production policy, trained from scratch.
    print("[2/3] Evaluating the production policy (PPO from scratch)")
    production_dir = run_models(PRIMARY_RUN)
    production_model = PPO.load(str(production_dir / "final_model.zip"), device="cpu")
    production_rows = evaluate_policy_on_scenarios(
        production_model, production_dir / "vecnormalize.pkl", scenarios, seed=42,
        env_kwargs={"w2": WEIGHT},
    )
    production = summarize("PPO from scratch", production_rows)
    print(f"      MRR {production['MRR_mean']:.1f} | energy {production['Energy_per_part']:.1f} kJ | "
          f"Ra viol (finished) {production['Ra_violation_finished_surface']*100:.1f}% | "
          f"success {production['Success']*100:.1f}%")

    # 3. The controller-initialized policy.
    print(f"[3/3] Training the controller-initialized policy ({timesteps:,} steps)")
    from experiments.train_rl import train
    init_dir = run_models(CONTROLLER_INITIALIZED_RUN)
    train(algo="PPO", timesteps=timesteps, seed=42, device="cpu", progress_bar=False,
          randomize=True, randomize_weight=True, run_name=CONTROLLER_INITIALIZED_RUN,
          init_from_controller=True, controller_episodes=controller_episodes)

    init_model = PPO.load(str(init_dir / "final_model.zip"), device="cpu")
    init_rows = evaluate_policy_on_scenarios(
        init_model, init_dir / "vecnormalize.pkl", scenarios, seed=42,
        env_kwargs={"w2": WEIGHT},
    )
    initialized = summarize("PPO initialized from the controller", init_rows)
    print(f"      MRR {initialized['MRR_mean']:.1f} | energy {initialized['Energy_per_part']:.1f} kJ | "
          f"Ra viol (finished) {initialized['Ra_violation_finished_surface']*100:.1f}% | "
          f"success {initialized['Success']*100:.1f}%")

    summary = pd.DataFrame([controller, production, initialized])
    summary.to_csv(out_dir / "controller_init_results.csv", index=False)

    print("\nController initialization experiment:")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
