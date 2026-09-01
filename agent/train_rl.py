import os
import sys
from pathlib import Path
import argparse

# CPU limits to avoid thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

gym.register(
    id="MillingDTEnv-v1",
    entry_point="env.milling_env:MillingMultipassEnv",
    max_episode_steps=50,
)

class ProgressCallback(BaseCallback):
    def __init__(self, progress_file, verbose=0):
        super().__init__(verbose)
        self.progress_file = progress_file

    def _on_step(self):
        # self.num_timesteps is the total across all envs
        if self.progress_file and self.num_timesteps % 1000 == 0:
            with open(self.progress_file, "w") as f:
                f.write(str(self.num_timesteps))
        return True

def train(algo="PPO", timesteps=1_000_000, seed=42, progress_file=None):
    log_dir = PROJECT_ROOT / "logs" / "tensorboard"
    models_dir = PROJECT_ROOT / "models" / f"{algo}_seed{seed}"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    train_env = make_vec_env("MillingDTEnv-v1", n_envs=4, seed=seed)
    
    # 1. Improvement: Reward & Observation Normalization
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # 3. Improvement: Larger network (256x256)
    policy_kwargs = dict(net_arch=[256, 256])
    
    if algo == "PPO":
        # 2. Improvement: Entropy tuning (ent_coef = 0.02)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.02, # Higher entropy for better exploration
            seed=seed,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs
        )
    elif algo == "SAC":
        # 4. Improvement: Soft Actor Critic
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            seed=seed,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f"Unknown algo {algo}")
        
    callbacks = []
    if progress_file:
        callbacks.append(ProgressCallback(progress_file))
        
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"{algo}_v3_seed{seed}"
    )
    
    # Save the final model and the VecNormalize stats
    model.save(str(models_dir / "final_model.zip"))
    train_env.save(str(models_dir / "vecnormalize.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_file", type=str, default=None)
    args = parser.parse_args()
    
    train(algo=args.algo, timesteps=args.steps, seed=args.seed, progress_file=args.progress_file)
