"""
agent/train_rl_conditioned.py

Trains the conditioned policy.

One PPO policy that is conditioned on the energy weight and on the material
context, trained over randomized scenarios. The weight is drawn per episode
and reaches the policy through the observation, so a single training run
covers the whole weight range instead of one run per weight.

Outputs:
    models/PPO_conditioned_seed42/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.train_rl import train
from env import CONDITIONED_ENV_ID
from config.paths import MODELS_DIR

ENV_ID = CONDITIONED_ENV_ID


def main():
    parser = argparse.ArgumentParser(description="Train the conditioned PPO policy")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    train(
        algo="PPO",
        timesteps=args.steps,
        seed=args.seed,
        device=args.device,
        progress_bar=False,
        env_id=ENV_ID,
        randomize=True,
        save_dir=str(MODELS_DIR / f"PPO_conditioned_seed{args.seed}"),
        run_name=f"PPO_conditioned_seed{args.seed}",
    )


if __name__ == "__main__":
    main()
