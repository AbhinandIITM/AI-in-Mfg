"""
experiments/batch_experiment.py

The batch against the part.

The controller in this project solves one part at a time. It has no notion of
the tool, which is the one resource a batch of parts shares: the flank wear
carries from one part to the next, and a tool that reaches its limit has to
be replaced. A policy that sees the tool life and the batch that is left can
trade the speed of one part against the tool life the rest of the batch needs.

Three arms, on the same batches:

1. the batch-aware policy, trained on the batch environment,
2. the per-part controller, applied part by part inside the batch,
3. the per-part policy, applied part by part inside the batch.

The second arm is what the controller does today, and the third is what the
learned policy does when it is not told about the tool. The difference
between the first and the third is what the batch context is worth.

Outputs:
    results/evaluation/batch_results.csv
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MillingDT.agent.controller_init import controller_action
from MillingDT.config import EVALUATION_DIR, PRIMARY_RUN, run_models
from MillingDT.env.milling_batch_env import MillingDTBatchEnv
from MillingDT.utils.seeding import set_seed

BATCH_ENV_ID = "MillingDTBatchEnv-v1"
BATCH_PARTS = 10
TOOL_CHANGE_COST = 500.0
WEIGHT = 0.4
BATCHES = 30
BATCH_RUN = "PPO_batch_seed42"


def make_batch_env(vec_norm_path=None, batch_parts=BATCH_PARTS):
    """A batch environment, with the normalization statistics of a policy."""
    env = gym.make(BATCH_ENV_ID, w2=WEIGHT, batch_parts=batch_parts,
                   tool_change_cost=TOOL_CHANGE_COST)
    vec = DummyVecEnv([lambda: env])
    if vec_norm_path is not None:
        vec = VecNormalize.load(str(vec_norm_path), vec)
        vec.training = False
        vec.norm_reward = False
    return vec


def load_observation_statistics(vec_norm_path):
    """The observation statistics of a policy, as a mean and a standard deviation.

    A per-part policy cannot be dropped into a batch through VecNormalize,
    because the batch observation has two more features than the policy was
    trained on. Its statistics are therefore applied to the eleven features
    it does read, by hand.
    """
    with open(vec_norm_path, "rb") as handle:
        stats = pickle.load(handle)
    rms = stats.obs_rms
    return rms.mean.astype(np.float32), np.sqrt(rms.var + 1e-8).astype(np.float32)


def summarize(name, rows):
    df = pd.DataFrame(rows)
    return {
        'arm': name,
        'mean_batch_reward': df['reward'].mean(),
        'reward_std': df['reward'].std(),
        'mean_part_reward': df['mean_part_reward'].mean(),
        'tool_changes': df['tool_changes'].mean(),
        'parts_done': df['parts_done'].mean(),
        'batch_T_total': df['batch_T_total'].mean(),
        'batch_E_total': df['batch_E_total'].mean(),
    }


def evaluate_batch_policy(model, vec_norm_path, per_part=False, seed=42,
                           batches=BATCHES):
    """The learned policy over a set of batches.

    With per_part the policy reads only the eleven features of the part and
    ignores the tool life and the batch that is left, which is what the
    production policy does when it is dropped into a batch.
    """
    rows = []
    mean, std = (None, None)
    if per_part:
        mean, std = load_observation_statistics(vec_norm_path)

    for index in range(batches):
        vec = make_batch_env(vec_norm_path if not per_part else None)
        obs = vec.reset()
        done = False
        total_reward = 0.0
        part_rewards = []
        steps = 0
        parts_before = 0
        tool_changes = 0

        while not done:
            if per_part:
                policy_obs = np.clip(
                    (obs[:, :11] - mean) / std, -10.0, 10.0)
            else:
                policy_obs = obs
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, done, info_list = vec.step(action)
            info = info_list[0]
            total_reward += float(reward[0])
            steps += 1
            # A part is over when the counter of finished parts moves. The
            # vectorized environment resets the batch when the episode ends,
            # so the counters are read from the payload as they arrive.
            if info['parts_done'] > parts_before:
                parts_before = info['parts_done']
                part_rewards.append(float(info.get('inner_reward', 0.0)))
            tool_changes = max(tool_changes, int(info.get('tool_changes', 0)))
            done = bool(done[0])

        rows.append({
            'reward': total_reward,
            'tool_changes': tool_changes,
            'parts_done': parts_before,
            'batch_T_total': float(info.get('batch_T_total', 0.0)),
            'batch_E_total': float(info.get('batch_E_total', 0.0)),
            'steps': steps,
            'mean_part_reward': float(np.mean(part_rewards)) if part_rewards else 0.0,
        })
        vec.close()
    return rows


def evaluate_controller(seed=42, batches=BATCHES):
    """The per-part controller over the same batches.

    It solves the part in front of it with the current wear and the current
    stock, and it knows nothing about the tool life left for the rest of the
    batch.
    """
    rows = []
    for index in range(batches):
        env = gym.make(BATCH_ENV_ID, w2=WEIGHT, batch_parts=BATCH_PARTS,
                       tool_change_cost=TOOL_CHANGE_COST).unwrapped
        env.reset(seed=seed + index)

        done = False
        total_reward = 0.0
        part_rewards = []
        steps = 0
        parts_before = 0
        while not done:
            action = controller_action(env.inner)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            if info['parts_done'] > parts_before:
                parts_before = info['parts_done']
                part_rewards.append(float(info.get('inner_reward', 0.0)))

        rows.append({
            'reward': total_reward,
            'tool_changes': env.tool_changes,
            'parts_done': env.parts_done,
            'batch_T_total': env.batch_T_total,
            'batch_E_total': env.batch_E_total,
            'steps': steps,
            'mean_part_reward': float(np.mean(part_rewards)) if part_rewards else 0.0,
        })
        env.close()
    return rows


def main():
    set_seed(42)
    records = []

    batch_dir = run_models(BATCH_RUN)
    if (batch_dir / "final_model.zip").exists():
        batch_model = PPO.load(str(batch_dir / "final_model.zip"), device="cpu")
        records.append(summarize(
            "PPO, batch aware",
            evaluate_batch_policy(batch_model, batch_dir / "vecnormalize.pkl")))
    else:
        print(f"No batch policy at {batch_dir}, skipping that arm.")

    part_dir = run_models(PRIMARY_RUN)
    if (part_dir / "final_model.zip").exists():
        part_model = PPO.load(str(part_dir / "final_model.zip"), device="cpu")
        records.append(summarize(
            "PPO, per part",
            evaluate_batch_policy(part_model, part_dir / "vecnormalize.pkl",
                                  per_part=True)))
    else:
        print(f"No per-part policy at {part_dir}, skipping that arm.")

    records.append(summarize("Receding-horizon controller, per part",
                              evaluate_controller()))

    summary = pd.DataFrame(records)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "batch_results.csv"
    summary.to_csv(out, index=False)

    print("\nBatch experiment:")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
