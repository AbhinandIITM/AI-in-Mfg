"""
experiments/regime_map.py

How noisy does the wear measurement have to be before the learned policy is
the better operator?

The robustness experiment compares two situations. This one maps the
boundary between them, so the answer is a number a plant can act on instead
of two cases.

The setup: the machine runs a material and cuts it with a tool whose flank
wear is measured with an instrument, not exactly. The reading carries a
Gaussian error of sigma, a fraction of the range of the wear and of the
remaining stock. The controller solves the next pass from that reading. The
policy reads the same noise on its observation.

Nothing is tilted towards the policy, and two points are deliberate:

    both arms are the general ones, neither is prepared for the noise. The
    policy that was trained with the noise exists, and it is reported as a
    third and clearly labelled arm, but it is not the headline comparison,
    because a controller can also be prepared for a noisy instrument by
    widening its safety margin, and that would be a design choice rather
    than a training one.

    the noise is tilted against the policy. The controller is given a noisy
    reading of the flank wear, which is the measurement it depends on. The
    policy gets noise on the flank wear AND on the remaining stock, so it is
    the one that is handicapped.

At zero noise this reproduces the nominal comparison of the robustness
experiment, which is the check that the two are consistent.

Outputs:
    results/evaluation/regime_map.csv
    results/plots/regime_map.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from experiments.robustness_experiment import (
    evaluate_controller,
    evaluate_policy,
    summarize,
    NOISY_RUN,
    WEIGHT,
)
from MillingDT.config import (
    EVALUATION_DIR,
    PLOTS_DIR,
    PRIMARY_RUN,
    run_models,
)
from MillingDT.config.scenarios import generate_random_configs
from MillingDT.utils.seeding import set_seed

SCENARIOS = 100
NOISES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def evaluate_arms(model, vec_norm_path, scenarios, noise, noisy_model=None):
    """The arms on the same scenarios, with the same measurement noise.

    The controller is given a noisy reading of the flank wear. The policy
    gets the noise on its observation, which covers the wear and the
    remaining stock.
    """
    controller_rows = evaluate_controller(
        scenarios, seed=42, weight=WEIGHT, obs_noise=noise,
        noise_rng=np.random.default_rng(7))
    controller = summarize("Receding-horizon controller", "nominal", controller_rows)

    policy_rows = evaluate_policy(
        model, vec_norm_path, scenarios, seed=42, weight=WEIGHT, obs_noise=noise)
    policy = summarize("PPO", "nominal", policy_rows)

    arms = [("Receding-horizon controller", controller), ("PPO", policy)]

    if noisy_model is not None:
        prepared_rows = evaluate_policy(
            noisy_model[0], noisy_model[1], scenarios, seed=42, weight=WEIGHT,
            obs_noise=noise)
        arms.append(("PPO trained for the noise",
                     summarize("PPO trained for the noise", "nominal", prepared_rows)))

    return arms


def main():
    set_seed(42)
    scenarios = generate_random_configs(SCENARIOS, seed=42)

    model_dir = run_models(PRIMARY_RUN)
    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    vec_norm_path = model_dir / "vecnormalize.pkl"

    noisy_model = None
    noisy_dir = run_models(NOISY_RUN)
    if (noisy_dir / "final_model.zip").exists():
        noisy_model = (PPO.load(str(noisy_dir / "final_model.zip"), device="cpu"),
                       noisy_dir / "vecnormalize.pkl")

    records = []
    for noise in NOISES:
        arms = evaluate_arms(model, vec_norm_path, scenarios, noise, noisy_model)
        for arm, row in arms:
            records.append({
                'noise': noise,
                'arm': arm,
                'mean_reward': row['mean_reward'],
                'reward_std': row['reward_std'],
                'MRR_mean': row['MRR_mean'],
                'Energy_per_part': row['Energy_per_part'],
                'Cycle_time': row['Cycle_time'],
                'Ra_violation_worst_pass': row['Ra_violation_worst_pass'],
                'Ra_violation_finished_surface': row['Ra_violation_finished_surface'],
                'Broken': row['Broken'],
                'Success': row['Success'],
            })
        line = ", ".join(f"{arm} {row['mean_reward']:.1f}" for arm, row in arms)
        print(f"  noise {noise:5.1%}: {line}")

    summary = pd.DataFrame(records)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "regime_map.csv"
    summary.to_csv(out, index=False)

    plot_regime_map(summary)

    print("\nThe boundary:")
    pivot = summary.pivot(index='noise', columns='arm', values='mean_reward')
    pivot['PPO ahead by'] = pivot['PPO'] - pivot['Receding-horizon controller']
    print(pivot.round(2).to_string())

    print(f"\nSaved to {out}")


def plot_regime_map(summary):
    """Where the learned policy overtakes the controller."""
    from viz.plotting_utils import set_style

    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = summary.pivot(index='noise', columns='arm', values='mean_reward')
    ax.plot(pivot.index * 100, pivot['Receding-horizon controller'],
            marker='o', color="#64748b", linewidth=2.4,
            label="Receding-horizon controller")
    ax.plot(pivot.index * 100, pivot['PPO'],
            marker='s', color="#ea580c", linewidth=2.4, label="PPO")
    if 'PPO trained for the noise' in pivot.columns:
        ax.plot(pivot.index * 100, pivot['PPO trained for the noise'],
                marker='^', color="#0d9488", linewidth=2.0, linestyle="--",
                label="PPO trained for the noise")

    ax.fill_between(pivot.index * 100, pivot['Receding-horizon controller'],
                    pivot['PPO'],
                    where=(pivot['PPO'] >= pivot['Receding-horizon controller']),
                    color="#ea580c", alpha=0.13, interpolate=True,
                    label="the learned policy is ahead")

    ax.set_title("How noisy the wear measurement has to be before the learned\n"
                 "policy is the better operator", fontweight="bold")
    ax.set_xlabel("Measurement noise on the wear and the remaining stock, "
                  "as a percent of their range")
    ax.set_ylabel("Mean episode reward over 100 scenarios")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best", framealpha=0.92)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "regime_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nRegime map saved to {out}")


if __name__ == "__main__":
    main()
