"""
sim/runner.py

Simulation runner engine for the CNC Milling Digital Twin Interactive Simulator.
Executes PPO policy inference and baseline controllers (Constant-High, Constant-Safe,
Rule-Based Decay, and NSGA-II Static) on MillingDTEnv with custom parameters.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env # Register MillingDTEnv-v1
from config.physics_constants import ALL_CONSTANTS

def run_simulation(controller_type: str, kwargs_env: dict, model_path: str = None) -> pd.DataFrame:
    """
    Runs a complete milling simulation for a specified controller type.

    Controllers:
        - 'PPO (DRL)'
        - 'Constant-High'
        - 'Constant-Safe'
        - 'Rule-Based Decay'
        - 'NSGA-II Static'
    """
    # Create environment instance
    env_inst = gym.make('MillingDTEnv-v1', **kwargs_env)

    # Load PPO model if applicable
    model = None
    vec_norm = None
    if controller_type == 'PPO (DRL)':
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "PPO_seed42"
        else:
            model_path = Path(model_path)

        model_file = model_path / "final_model.zip"
        stats_file = model_path / "vecnormalize.pkl"

        if model_file.exists():
            model = PPO.load(str(model_file), device="cpu")
            if stats_file.exists():
                # Wrap in dummy env for VecNormalize stats loading
                dummy_env = DummyVecEnv([lambda: gym.make('MillingDTEnv-v1', **kwargs_env)])
                vec_norm = VecNormalize.load(str(stats_file), dummy_env)
                vec_norm.training = False
                vec_norm.norm_reward = False

    obs, info = env_inst.reset()

    records = []

    # Record initial state (pass 0)
    records.append({
        'pass': 0,
        'controller': controller_type,
        'kappa_rem': kwargs_env.get('kappa_total', 5.0),
        'a_p': 0.0,
        'f_r': 0.0,
        'n': 0.0,
        'Ra': 0.0,
        'MRR': 0.0,
        'VB': kwargs_env.get('VB_0', 0.0),
        'Energy': 0.0,
        'reward': 0.0,
        'broken': False,
        'is_finishing': False
    })

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 200:
        step_count += 1

        # Determine action based on controller type
        if controller_type == 'PPO (DRL)' and model is not None:
            if vec_norm is not None:
                norm_obs = vec_norm.normalize_obs(obs)
                action, _ = model.predict(norm_obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
        elif controller_type == 'Constant-High':
            # Max action [1, 1, 1]
            action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif controller_type == 'Constant-Safe':
            # Min action [-1, -1, -1]
            action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        elif controller_type == 'Rule-Based Decay':
            # Decrease spindle speed gradually, keep moderate feed and depth
            n_progress = max(0.0, 1.0 - (step_count * 0.05))
            action = np.array([n_progress * 2.0 - 1.0, 0.2, 0.5], dtype=np.float32)
        elif controller_type == 'NSGA-II Static':
            # Balanced static action near optimal Pareto knee
            action = np.array([0.2, 0.0, 0.4], dtype=np.float32)
        else:
            action = env_inst.action_space.sample()

        obs, reward, terminated, truncated, info = env_inst.step(action)

        record = {
            'pass': info['pass'],
            'controller': controller_type,
            'kappa_rem': info['kappa_rem'],
            'a_p': info['a_p'],
            'f_r': info['f_r'],
            'n': info['n'],
            'Ra': info['Ra'],
            'MRR': info['MRR'],
            'VB': info['VB'],
            'Energy': info['Energy'],
            'reward': reward,
            'broken': info['broken'],
            'is_finishing': info['is_finishing']
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

def run_all_comparisons(kwargs_env: dict) -> dict:
    """
    Runs PPO and all baselines on the same environment configuration.
    """
    controllers = [
        'PPO (DRL)',
        'Constant-High',
        'Constant-Safe',
        'Rule-Based Decay',
        'NSGA-II Static'
    ]

    results = {}
    for ctrl in controllers:
        results[ctrl] = run_simulation(ctrl, kwargs_env)

    return results
