import os
import sys
import time
from pathlib import Path
import argparse

# CPU limits to avoid thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import registration and seeding
import env
from utils.seeding import set_seed

class ProgressQueueCallback(BaseCallback):
    """Reports training progress to the supervising process over a Queue.

    The heartbeat is what tells agent/run_experiments.py that a worker is
    still alive. It must never fail silently: a dropped heartbeat looks
    identical to a hung worker, and the stall detector terminates the job.
    """

    REPORT_INTERVAL = 1000

    def __init__(self, queue=None, verbose=0):
        super().__init__(verbose)
        self.queue = queue

    def _on_step(self):
        if self.queue and self.num_timesteps % self.REPORT_INTERVAL == 0:
            rewards = self.locals.get("rewards", [0.0])
            try:
                self.queue.put_nowait({
                    'timesteps': self.num_timesteps,
                    'reward': float(np.mean(rewards)),
                    'timestamp': time.time()
                })
            except Exception as exc:
                self._log(f"Progress heartbeat dropped at {self.num_timesteps} steps: {exc!r}")
        return True

    def _log(self, message):
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.warning(message)
        else:
            print(f"[WARN] {message}", flush=True)

def train(algo="PPO", timesteps=1_000_000, seed=42, queue=None, device="auto", progress_bar=None, save_dir=None, env_kwargs=None, tb_log_name=None, randomize=False, policy=None, env_id="MillingDTEnv-v1", init_weights=None):
    """Train a PPO policy and save the policy plus its VecNormalize statistics.

    Args:
        env_kwargs: forwarded to MillingDTEnv (reward weights, physics overrides).
        tb_log_name: TensorBoard run name. Pass a unique name per run, otherwise
            consecutive runs append to the same TensorBoard directory.
        randomize: draw a new machining scenario for every episode.
        policy: an existing policy to warm start from.
        env_id: which environment registration to train against.
        init_weights: callable that receives the fresh model before training,
            used to seed a policy with pre-trained weights.
    """
    # Set seed
    set_seed(seed)

    env_kwargs = dict(env_kwargs or {})
    if randomize:
        env_kwargs['randomize'] = True

    log_dir = PROJECT_ROOT / "results" / "tensorboard"
    if save_dir is not None:
        models_dir = Path(save_dir)
    else:
        models_dir = PROJECT_ROOT / "models" / f"PPO_seed{seed}"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect or use user device
    if device == "auto":
        # For PPO on MLP, CPU is typically faster and avoids unnecessary host-tXdevice kernel overhead
        device = "cpu"

    # Default progress_bar: True for direct interactive runs, False when run via worker queue
    if progress_bar is None:
        progress_bar = (queue is None)

    # Create environments
    train_env = make_vec_env(env_id, n_envs=4, seed=seed, env_kwargs=env_kwargs)

    # Reward & Observation Normalization
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # Larger network (256x256)
    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
        policy if policy is not None else "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.02,
        seed=seed,
        tensorboard_log=str(log_dir),
        policy_kwargs=policy_kwargs,
        device=device
    )

    callbacks = []
    if queue:
        callbacks.append(ProgressQueueCallback(queue))

    if init_weights is not None:
        init_weights(model)

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name or f"PPO_primary_seed{seed}",
        progress_bar=progress_bar
    )

    # Save the final model and the VecNormalize stats
    final_model_path = models_dir / "final_model.zip"
    vec_norm_path = models_dir / "vecnormalize.pkl"
    model.save(str(final_model_path))
    train_env.save(str(vec_norm_path))

    if queue is None:
        print("\n" + "=" * 65)
        print("  [OK] Training Complete!")
        print("=" * 65)
        print(f"  - Model Policy:        {final_model_path}")
        print(f"  - Normalization Stats: {vec_norm_path}")
        print(f"  - TensorBoard Logs:    {log_dir}")
        print("=" * 65)
        print("  Next Steps:")
        print("    1. Evaluate model:   python agent/evaluate.py")
        print("    2. View curves:      tensorboard --logdir results/tensorboard")
        print("    3. Plot curves:      python viz/plot_results.py")
        print("=" * 65 + "\n")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--randomize", action="store_true",
                        help="draw a new machining scenario for every episode")
    args = parser.parse_args()

    train(algo=args.algo, timesteps=args.steps, seed=args.seed, device=args.device,
          randomize=args.randomize)


