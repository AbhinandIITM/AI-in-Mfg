"""
experiments/train_rl.py

The one training entry point of the project.

The environment carries the energy weight and the material context in its
observation, so the same script trains every policy in the project:

    # the production policy: the weight is drawn per episode, so one model
    # covers the whole weight range and the whole material distribution
    python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight

    # a policy for one point of the trade-off (the ablation study)
    python -m experiments.train_rl --steps 150000 --seed 42 --randomize --w2 0.6

    # a policy for one material (the transfer experiment)
    python -m experiments.train_rl --steps 300000 --seed 42

Every run writes into its own directory under results/:
results/run_<run_name>/models holds the policy and its normalization
statistics, results/run_<run_name>/log holds the training curve.
"""

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
from stable_baselines3.common.logger import configure

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import registration and seeding
import MillingDT.env
from MillingDT.env import ENV_ID
from MillingDT.config import run_log, run_models
from MillingDT.utils.seeding import set_seed

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

def train(algo="PPO", timesteps=1_000_000, seed=42, queue=None, device="auto", progress_bar=None,
          env_kwargs=None, run_name=None, randomize=False,
          randomize_weight=False, w2=None, policy=None, env_id=ENV_ID, init_weights=None,
          init_from_controller=False, controller_episodes=2000):
    """Train a PPO policy and save the policy plus its VecNormalize statistics.

    Args:
        env_kwargs: forwarded to MillingDTEnv (physics overrides, a pinned w2).
        run_name: name of the run. The policy and the training curve are
            written to results/run_<run_name>/models and .../log.
        randomize: draw a new machining scenario for every episode.
        randomize_weight: draw the energy weight for every episode.
        w2: the pinned energy weight of the task. w1 = 1 - w2.
        policy: an existing policy to warm start from.
        env_id: which environment registration to train against.
        init_weights: callable that receives the fresh model before training,
            used to seed a policy with pre-trained weights.
        init_from_controller: start the actor from the receding-horizon
            controller, by recording what the controller does and fitting a
            supervised actor on those trajectories.
        controller_episodes: how many controller episodes to record for the
            initialization.
    """
    # Set seed
    set_seed(seed)

    env_kwargs = dict(env_kwargs or {})
    if randomize:
        env_kwargs['randomize'] = True
    if randomize_weight:
        env_kwargs['randomize_weight'] = True
    if w2 is not None:
        env_kwargs['w2'] = float(w2)

    run_name = run_name or f"PPO_seed{seed}"
    models_dir = run_models(run_name)
    models_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = run_log(run_name)
    curve_dir.mkdir(parents=True, exist_ok=True)

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
        policy_kwargs=policy_kwargs,
        device=device
    )

    callbacks = []
    if queue:
        callbacks.append(ProgressQueueCallback(queue))

    if init_weights is not None:
        init_weights(model)
    elif init_from_controller:
        # Start the actor from what the receding-horizon controller does, so
        # that reinforcement learning begins from a feasible behaviour
        # instead of from noise.
        from MillingDT.agent.controller_init import (
            collect_controller_dataset,
            copy_into_policy,
            fit_cloned_actor,
        )
        print(f"  Recording {controller_episodes} controller episodes...")
        X, Y = collect_controller_dataset(episodes=controller_episodes, seed=seed,
                                          randomize=randomize,
                                          randomize_weight=randomize_weight)
        print(f"  Fitting the cloned actor on {len(X)} transitions...")
        actor = fit_cloned_actor(X, Y, seed=seed)
        copy_into_policy(model, actor)
        print("  Actor initialized from the controller.")

    # The training curve is written by Stable-Baselines3's own CSV logger, so
    # the artifact is the standard progress.csv with the rollout and training
    # statistics, and no extra dependency is needed to read it back.
    model.set_logger(configure(folder=str(curve_dir), format_strings=["csv"]))

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
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
        print(f"  - Training Curve:      {curve_dir / 'progress.csv'}")
        print("=" * 65)
        print("  Next Steps:")
        print("    1. Evaluate model:   python -m experiments.evaluate")
        print("    2. Plot curves:      python -m viz.plot_results")
        print("    3. Rebuild tables:   python -m experiments.summarize_results")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--randomize", action="store_true",
                        help="draw a new machining scenario for every episode")
    parser.add_argument("--randomize-weight", action="store_true",
                        help="draw the energy weight for every episode")
    parser.add_argument("--w2", type=float, default=None,
                        help="pin the energy weight of the task (w1 = 1 - w2)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="name of the run directory under results/")
    parser.add_argument("--init-from-controller", action="store_true",
                        help="start the actor from the receding-horizon controller")
    parser.add_argument("--controller-episodes", type=int, default=2000,
                        help="controller episodes recorded for the initialization")
    args = parser.parse_args()

    train(algo=args.algo, timesteps=args.steps, seed=args.seed, device=args.device,
          randomize=args.randomize, randomize_weight=args.randomize_weight, w2=args.w2,
          run_name=args.run_name, init_from_controller=args.init_from_controller,
          controller_episodes=args.controller_episodes)
