"""
experiments/transfer_experiment.py

How much of a new material does a policy that was trained on another one
already know?

The experiment has three arms on the same target task (a harder material on
the same machine):

1. the source policy, zero shot,
2. the source policy, fine tuned on the target for a given budget,
3. a policy trained on the target from scratch for the same budget.

The comparison of 2 against 3 says how much of the new task the source policy
already covered, and the curve over the fine tuning budget says how fast it
adapts.

Outputs:
    results/evaluation/transfer_results.csv
    results/evaluation/transfer_curve.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import MillingDT.env # Triggers registration
from MillingDT.env import ENV_ID
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MillingDT.config import (
    ALL_CONSTANTS,
    EVALUATION_DIR,
    PRIMARY_RUN,
    TRANSFER_SCRATCH_RUN,
    TRANSFER_SOURCE_RUN,
    run_models,
)
from MillingDT.config.scenarios import generate_random_configs
from MillingDT.utils.seeding import set_seed

# The target material: a harder alloy and a faster wearing tool, on the same
# machine. The source policy has never seen any of it.
TARGET_SCENARIO = {
    "k1": ALL_CONSTANTS['surface_roughness']['k1'] * 1.25,
    "k2": ALL_CONSTANTS['surface_roughness']['k2'] * 1.30,
    "C_wear": ALL_CONSTANTS['tool_wear']['C_wear'] * 2.0,
}


def evaluate_on_target(model, vec_norm_path, scenarios, env_kwargs=None, seed=42):
    """Mean episode reward and constraint behaviour on the target task."""
    env_kwargs = dict(env_kwargs or {})
    rewards, violations, final_violations, successes = [], [], [], []

    for idx, (name, scenario) in enumerate(scenarios.items()):
        # The target material overrides the scenario, it does not duplicate it.
        env_kwargs_full = {**scenario, **env_kwargs}
        eval_env = gym.make(ENV_ID, **env_kwargs_full)
        eval_env.reset(seed=seed + idx)
        vec_env = DummyVecEnv([lambda: eval_env])
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        max_ra = 0.0
        final_ra = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            info = info[0]
            total_reward += float(reward[0])
            max_ra = max(max_ra, info['Ra'])
            final_ra = info['Ra']
            done = bool(done[0])

        rewards.append(total_reward)
        violations.append(max_ra > ALL_CONSTANTS['engineering']['Ra_max'])
        final_violations.append(final_ra > ALL_CONSTANTS['engineering']['Ra_max'])
        successes.append(bool(info.get('is_finishing', False)) and not info.get('broken', False))

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'ra_violation_rate': float(np.mean(violations)),
        'ra_final_violation_rate': float(np.mean(final_violations)),
        'success_rate': float(np.mean(successes)),
        'episodes': len(rewards),
    }


def train_on(env_kwargs, timesteps, seed, run_name, env_id=ENV_ID):
    """Train a policy from scratch on a given task."""
    from experiments.train_rl import train

    train(
        algo="PPO", timesteps=timesteps, seed=seed, device="cpu",
        progress_bar=False, env_kwargs=env_kwargs, run_name=run_name,
    )
    return run_models(run_name)


def main():
    source_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    adapt_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    chunks = 5

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    scenarios = generate_random_configs(30, seed=42)

    # 1. The source policy: trained on the nominal material.
    source_dir = run_models(TRANSFER_SOURCE_RUN)
    if not (source_dir / "final_model.zip").exists():
        print(f"[1/4] Training the source policy on the nominal material ({source_steps:,} steps)")
        train_on({}, source_steps, 42, TRANSFER_SOURCE_RUN)
    else:
        print(f"[1/4] Reusing the source policy at {source_dir}")

    source_model = PPO.load(str(source_dir / "final_model.zip"), device="cpu")
    source_vec = source_dir / "vecnormalize.pkl"

    # 2. Zero shot on the target.
    print("[2/4] Evaluating the source policy on the target material")
    zero_shot = evaluate_on_target(source_model, source_vec, scenarios,
                                    env_kwargs=TARGET_SCENARIO)
    print(f"      mean reward {zero_shot['mean_reward']:.1f} | "
          f"Ra violations {zero_shot['ra_violation_rate']*100:.1f}% | "
          f"success {zero_shot['success_rate']*100:.1f}%")

    # 3. Fine tune on the target, evaluating after every chunk.
    print(f"[3/4] Fine tuning on the target material ({adapt_budget:,} steps in {chunks} chunks)")
    curve = [{'stage': 'source', 'steps_on_target': 0, **zero_shot}]

    model = PPO.load(str(source_dir / "final_model.zip"), device="cpu")
    env = make_vec_env(ENV_ID, n_envs=4, seed=42, env_kwargs=TARGET_SCENARIO)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    model.set_env(env)

    chunk_steps = adapt_budget // chunks
    spent = 0
    for i in range(chunks):
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, progress_bar=False)
        spent += chunk_steps
        vec_path = source_dir / f"finetune_{spent}.pkl"
        env.save(str(vec_path))
        metrics = evaluate_on_target(model, vec_path, scenarios, env_kwargs=TARGET_SCENARIO)
        curve.append({'stage': 'finetuned', 'steps_on_target': spent, **metrics})
        print(f"      {spent:>7,} steps on target: mean reward {metrics['mean_reward']:.1f} | "
              f"Ra violations {metrics['ra_violation_rate']*100:.1f}% | "
              f"success {metrics['success_rate']*100:.1f}%")

    # 4. From scratch on the target, same budget.
    print(f"[4/5] Training from scratch on the target material ({adapt_budget:,} steps)")
    scratch_dir = run_models(TRANSFER_SCRATCH_RUN)
    train_on(TARGET_SCENARIO, adapt_budget, 42, TRANSFER_SCRATCH_RUN)
    scratch_model = PPO.load(str(scratch_dir / "final_model.zip"), device="cpu")
    scratch = evaluate_on_target(scratch_model, scratch_dir / "vecnormalize.pkl", scenarios,
                                  env_kwargs=TARGET_SCENARIO)
    print(f"      mean reward {scratch['mean_reward']:.1f} | "
          f"Ra violations {scratch['ra_violation_rate']*100:.1f}% | "
          f"success {scratch['success_rate']*100:.1f}%")

    curve.append({'stage': 'from_scratch', 'steps_on_target': adapt_budget, **scratch})

    # 5. The production policy, zero shot. It was trained across a
    # distribution of materials with the material context in its
    # observation, so the target material is inside its training
    # distribution and needs no adaptation at all.
    print("[5/5] Evaluating the production policy on the target material")
    production = None
    production_dir = run_models(PRIMARY_RUN)
    if (production_dir / "final_model.zip").exists():
        production_model = PPO.load(str(production_dir / "final_model.zip"), device="cpu")
        production = evaluate_on_target(production_model, production_dir / "vecnormalize.pkl",
                                        scenarios, env_kwargs=TARGET_SCENARIO)
        print(f"      mean reward {production['mean_reward']:.1f} | "
              f"Ra violations {production['ra_violation_rate']*100:.1f}% | "
              f"success {production['success_rate']*100:.1f}%")
        curve.append({'stage': 'production_zero_shot', 'steps_on_target': 0, **production})
    else:
        print(f"      no production policy at {production_dir}, skipping this arm")

    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(EVALUATION_DIR / "transfer_curve.csv", index=False)

    final_finetuned = curve[-2]
    summary = {
        'source_steps': source_steps,
        'adapt_budget': adapt_budget,
        'zero_shot_reward': zero_shot['mean_reward'],
        'finetuned_reward': final_finetuned['mean_reward'],
        'from_scratch_reward': scratch['mean_reward'],
        'zero_shot_ra_violation': zero_shot['ra_violation_rate'],
        'finetuned_ra_violation': final_finetuned['ra_violation_rate'],
        'from_scratch_ra_violation': scratch['ra_violation_rate'],
        'zero_shot_ra_final_violation': zero_shot['ra_final_violation_rate'],
        'finetuned_ra_final_violation': final_finetuned['ra_final_violation_rate'],
        'from_scratch_ra_final_violation': scratch['ra_final_violation_rate'],
        'zero_shot_success': zero_shot['success_rate'],
        'finetuned_success': final_finetuned['success_rate'],
        'from_scratch_success': scratch['success_rate'],
    }
    if production is not None:
        summary['production_reward'] = production['mean_reward']
        summary['production_ra_violation'] = production['ra_violation_rate']
        summary['production_ra_final_violation'] = production['ra_final_violation_rate']
        summary['production_success'] = production['success_rate']
    summary = pd.DataFrame([summary])
    summary.to_csv(EVALUATION_DIR / "transfer_results.csv", index=False)

    print("\nTransfer summary:")
    print(summary.round(3).T.to_string())
    print(f"\nSaved to {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
