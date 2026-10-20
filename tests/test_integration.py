"""Integration tests: the pieces have to work together, not just alone."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from env import ENV_ID
from utils.seeding import set_seed


def test_training_to_evaluation_roundtrip(tmp_path):
    """Train a policy for a few thousand steps and evaluate it.

    The unit tests cover the environment and the callback in isolation. This
    covers the path that actually runs in the pipeline: train, save the
    policy and its normalization statistics, load them again, and roll the
    policy out over randomized scenarios.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from agent.evaluate import evaluate_policy_on_scenarios
    from config.scenarios import generate_random_configs

    set_seed(7)
    train_env = make_vec_env(ENV_ID, n_envs=2, seed=7, env_kwargs={"randomize": True})
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=32,
                seed=7, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=512)

    stats = model.get_env()
    vec_path = tmp_path / "vecnormalize.pkl"
    stats.save(str(vec_path))

    scenarios = generate_random_configs(3, seed=7)
    rows = evaluate_policy_on_scenarios(model, vec_path, scenarios, seed=7)

    assert len(rows) == 3
    df = pd.DataFrame(rows)
    assert df["MRR_mean"].gt(0).all()
    assert df["Energy_total"].gt(0).all()
    assert df["T_total"].gt(0).all()
    assert df["passes"].ge(1).all()
    # A policy that never finishes the part would score no completion at all.
    assert df["Success"].sum() > 0

    model.save(str(tmp_path / "model.zip"))
    stats.close()


def test_there_is_one_milling_registration():
    """The two registrations were merged into a single environment.

    The energy weight and the material moved into the observation, so a
    task with a pinned weight is the same environment with a constant in the
    context, not a second registration.
    """
    import gymnasium as gym

    assert env.ENV_ID == "MillingDTEnv-v1"
    milling_ids = sorted(i for i in gym.registry if i.startswith("MillingDTEnv"))
    assert milling_ids == [env.ENV_ID]


def test_registered_environment_observation_carries_the_context():
    """The registration must expose the weight and the material context."""
    import gymnasium as gym

    env_inst = gym.make(env.ENV_ID).unwrapped
    obs, _ = env_inst.reset(seed=3)

    assert obs.shape == (11,)
    # The last three features are the context: the energy weight and the
    # material coefficients, all normalized onto [0, 1].
    assert 0.0 <= obs[8] <= 1.0
    assert 0.0 <= obs[9] <= 1.0
    assert 0.0 <= obs[10] <= 1.0
    # The nominal material sits in the middle of the bands the scenarios span.
    assert obs[9] == pytest.approx(0.5)
    assert obs[10] == pytest.approx(0.5)
    env_inst.close()


def test_one_policy_answers_a_pinned_weight(tmp_path):
    """The weight is a query of the policy, not a retrain.

    The policy is trained once over the whole weight range and is then asked
    for a pinned weight. The weight is part of the observation, so the same
    network and the same normalization statistics serve both.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from agent.evaluate import evaluate_policy_on_scenarios
    from config.scenarios import generate_random_configs

    set_seed(9)
    train_env = make_vec_env(ENV_ID, n_envs=2, seed=9,
                              env_kwargs={"randomize": True, "randomize_weight": True})
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=128, batch_size=32,
                seed=9, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=256)

    stats = model.get_env()
    vec_path = tmp_path / "vecnormalize_weight.pkl"
    stats.save(str(vec_path))

    scenarios = generate_random_configs(2, seed=9)
    rows = evaluate_policy_on_scenarios(model, vec_path, scenarios, seed=9,
                                        env_kwargs={"w2": 0.9})

    assert len(rows) == 2
    df = pd.DataFrame(rows)
    assert df["MRR_mean"].gt(0).all()
    assert df["Energy_total"].gt(0).all()
    assert df["passes"].ge(1).all()

    # The policy was asked for the pinned weight, and the environment told
    # it so in the observation.
    probe = gym.make(ENV_ID, w2=0.9).unwrapped
    obs, _ = probe.reset(seed=9)
    assert obs[8] == pytest.approx(0.9)
    probe.close()

    stats.close()


def test_baseline_comparison_writes_its_artifact():
    """The baseline script must produce a table with a row per policy."""
    from baselines.run_baselines import evaluate_all_baselines

    # Two episodes per policy is enough to check the plumbing.
    results_dir = PROJECT_ROOT / "results"
    original = (results_dir / "baseline_results_v5.csv")
    backup = original.read_bytes() if original.exists() else None
    try:
        evaluate_all_baselines()
        assert original.exists()
        df = pd.read_csv(original)
        assert set(df["policy"]) >= {"constant_high", "constant_safe",
                                     "rule_based", "nsga2", "receding_horizon"}
        assert df["total_reward"].notna().all()
    finally:
        if backup is not None:
            original.write_bytes(backup)
