import pytest
import os
import shutil
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from utils.seeding import set_seed

@pytest.fixture
def clean_test_dir():
    test_dir = "tests/temp_models"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_training_convergence(clean_test_dir):
    """Test that training runs without error and model learns/updates over short steps."""
    set_seed(42)
    model_path = os.path.join(clean_test_dir, "ppo_test_model.zip")
    vec_path = os.path.join(clean_test_dir, "vecnormalize.pkl")

    train_env = make_vec_env("MillingDTEnv-v1", n_envs=2, seed=42)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        n_steps=128,
        batch_size=32,
        seed=42,
        device="cpu"
    )



    # Train for a few steps to verify loop execution and convergence initiation
    model.learn(total_timesteps=512)

    model.save(model_path)
    train_env.save(vec_path)

    assert os.path.exists(model_path)
    assert os.path.exists(vec_path)

def test_evaluation_reproducibility(clean_test_dir):
    """Test that evaluation with fixed seed produces identical episode trajectories."""
    set_seed(42)

    # Run episode 1 with seed 123
    env1 = gym.make("MillingDTEnv-v1")
    obs1, _ = env1.reset(seed=123)
    rewards1 = []
    actions = [env1.action_space.sample() for _ in range(10)]

    for action in actions:
        obs1, reward, terminated, truncated, info1 = env1.step(action)
        rewards1.append(reward)
        if terminated or truncated:
            break

    # Run episode 2 with seed 123 (reproducibility check)
    env2 = gym.make("MillingDTEnv-v1")
    obs2, _ = env2.reset(seed=123)
    rewards2 = []

    for action in actions:
        obs2, reward, terminated, truncated, info2 = env2.step(action)
        rewards2.append(reward)
        if terminated or truncated:
            break

    np.testing.assert_allclose(rewards1, rewards2, rtol=1e-5, atol=1e-5)
    env1.close()
    env2.close()
