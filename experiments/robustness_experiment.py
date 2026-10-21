"""
experiments/robustness_experiment.py

Where the learned policy and the classical controller separate.

The controller solves every pass with the exact model, the exact state and the
material in its hand. The learned policy reads the material in its context and
was trained across a distribution of scenarios. Two situations break the
controller's advantage, and this script measures both.

1. The nominal machine. The reference comparison, where the controller has
   every advantage.

2. A noisy measurement of the wear. The flank wear and the remaining stock are
   measured with an instrument, not exactly, and the reading carries a
   Gaussian error of sigma, a fraction of their range. The controller solves
   the next pass from that reading, so a wrong reading becomes a wrong plan.
   The policy reads the same noise on its observation, and a perturbed
   observation perturbs its action rather than invalidating its plan.

Nothing is tilted towards the policy, and two points are deliberate. Both arms
are the general ones, neither is trained for the noise, and the policy that was
trained with the noise is reported as a third and clearly labelled arm, because
a controller can also be prepared for a noisy instrument by widening its
safety margin, and that would be a design choice rather than a training one.
And the noise is tilted against the policy: the controller is given a noisy
reading of the flank wear, the measurement it depends on, while the policy
receives noise on the flank wear AND on the remaining stock.

Outputs:
    results/evaluation/robustness_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO

from MillingDT.agent.controller_init import controller_action
from experiments.evaluate import evaluate_policy_on_scenarios
from MillingDT.config import (
    EVALUATION_DIR,
    PRIMARY_RUN,
    run_models,
)
from MillingDT.config.scenarios import generate_random_configs
from MillingDT.utils.seeding import set_seed

WEIGHT = 0.4
SCENARIOS = 100

# The measurement noise on the wear and the remaining stock, as a fraction of
# their range.
NOISE = 0.05
NOISY_RUN = "PPO_noisystate_seed42"


def evaluate_controller(scenarios, seed=42, weight=WEIGHT, belief=None,
                        beliefs=None, obs_noise=0.0, noise_rng=None):
    """Roll the controller out, with the same measurement noise the policy sees.

    With obs_noise the controller is given a noisy reading of the flank wear,
    the measurement it depends on. The remaining stock it knows exactly.

    The belief is what the controller thinks the material is. The beliefs
    argument carries one belief per scenario, which is what a setup with a
    miscalibrated material model has.
    """
    rows = []
    for idx, (name, scenario) in enumerate(scenarios.items()):
        eval_env = gym.make("MillingDTEnv-v1", w2=weight, obs_noise=obs_noise,
                            **scenario)
        eval_env.reset(seed=seed + idx)
        unwrapped = eval_env.unwrapped

        obs, _ = eval_env.reset(seed=seed + idx)
        done = False
        ep_MRR, ep_Energy, ep_Ra, ep_VB = [], [], [], []
        ep_n, ep_f, ep_ap = [], [], []
        info = {}
        total_reward = 0.0
        scenario_belief = belief
        if beliefs and name in beliefs:
            scenario_belief = beliefs[name]
        while not done:
            measured_vb = unwrapped.VB
            if obs_noise and noise_rng is not None:
                measured_vb += noise_rng.normal(0.0, obs_noise * unwrapped.VB_max)
            action = controller_action(unwrapped, belief=scenario_belief,
                                      vb=measured_vb)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            ep_MRR.append(info['MRR'])
            ep_Energy.append(info['Energy'])
            ep_Ra.append(info['Ra'])
            ep_VB.append(info['VB'])
            ep_n.append(info['n'])
            ep_f.append(info['f_r'])
            ep_ap.append(info['a_p'])

        rows.append({
            'config': name,
            'reward': total_reward,
            'MRR_mean': float(np.mean(ep_MRR)),
            'Energy_total': float(np.sum(ep_Energy)),
            'Ra_max': float(np.max(ep_Ra)),
            'Ra_final': float(ep_Ra[-1]),
            'VB_max': float(np.max(ep_VB)),
            'T_total': float(info.get('T_total', 0.0)),
            'n_mean': float(np.mean(ep_n)),
            'f_r_mean': float(np.mean(ep_f)),
            'a_p_mean': float(np.mean(ep_ap)),
            'passes': len(ep_MRR),
            'Ra_violation': int(np.max(ep_Ra) > 1.6),
            'Ra_final_violation': int(ep_Ra[-1] > 1.6),
            'VB_violation': int(np.max(ep_VB) > 0.3),
            'Broken': int(bool(info.get('broken', False))),
            'Success': int(bool(info.get('is_finishing', False))
                           and not info.get('broken', False)),
        })
        eval_env.close()

    return rows


def evaluate_policy(model, vec_norm_path, scenarios, seed=42, weight=WEIGHT,
                    belief=None, beliefs=None, obs_noise=0.0):
    """The learned policy on the same scenarios, with the same noise.

    The belief is what the setup believes the material is, and it is what the
    policy reads in its context. The beliefs argument carries one belief per
    scenario.
    """
    env_kwargs = {}
    if belief is not None:
        env_kwargs['belief'] = belief
    if obs_noise:
        env_kwargs['obs_noise'] = obs_noise

    configs = {}
    for name, scenario in scenarios.items():
        per_scenario = dict(env_kwargs)
        if beliefs and name in beliefs:
            per_scenario['belief'] = beliefs[name]
        configs[name] = {**scenario, **per_scenario}

    return evaluate_policy_on_scenarios(
        model, vec_norm_path, configs, seed=seed,
        env_kwargs={"w2": weight})


def summarize(name, regime, rows):
    df = pd.DataFrame(rows)
    return {
        'regime': regime,
        'arm': name,
        'mean_reward': df['reward'].mean(),
        'reward_std': df['reward'].std(),
        'MRR_mean': df['MRR_mean'].mean(),
        'MRR_std': df['MRR_mean'].std(),
        'Energy_per_part': df['Energy_total'].mean(),
        'Cycle_time': df['T_total'].mean(),
        'Ra_violation_worst_pass': df['Ra_violation'].mean(),
        'Ra_violation_finished_surface': df['Ra_final_violation'].mean(),
        'VB_violation': df['VB_violation'].mean(),
        'Broken': df['Broken'].mean(),
        'Success': df['Success'].mean(),
    }


def main():
    set_seed(42)
    scenarios = generate_random_configs(SCENARIOS, seed=42)

    model_dir = run_models(PRIMARY_RUN)
    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")
    vec_norm_path = model_dir / "vecnormalize.pkl"

    records = []

    # 1. The nominal machine. The controller knows the material, the policy is
    # told the material, and the reference comparison.
    records.append(summarize(
        "Receding-horizon controller", "nominal",
        evaluate_controller(scenarios)))
    records.append(summarize(
        "PPO", "nominal",
        evaluate_policy(model, vec_norm_path, scenarios)))

    # 2. A noisy measurement of the wear. The controller trusts the reading
    # it gets, the policy reads the same noise on its observation.
    noisy_dir = run_models(NOISY_RUN)
    if (noisy_dir / "final_model.zip").exists():
        noisy_model = PPO.load(str(noisy_dir / "final_model.zip"), device="cpu")
        records.append(summarize(
            "Receding-horizon controller", "noisy wear measurement",
            evaluate_controller(scenarios, obs_noise=NOISE,
                                noise_rng=np.random.default_rng(7))))
        # The general policy faces the same noise. It is the headline arm of
        # this regime, because it is the one that was not prepared for the
        # noise; the policy that was trained for it is reported next to it.
        records.append(summarize(
            "PPO", "noisy wear measurement",
            evaluate_policy(model, vec_norm_path, scenarios, obs_noise=NOISE)))
        records.append(summarize(
            "PPO trained with the noise", "noisy wear measurement",
            evaluate_policy(noisy_model, noisy_dir / "vecnormalize.pkl",
                            scenarios, obs_noise=NOISE)))
    else:
        print(f"No policy trained with the noise at {noisy_dir}, skipping that regime.")

    summary = pd.DataFrame(records)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "robustness_results.csv"
    summary.to_csv(out, index=False)

    print("\nRobustness experiment:")
    print(summary[['regime', 'arm', 'mean_reward', 'reward_std', 'MRR_mean',
                   'Energy_per_part', 'Cycle_time', 'Broken', 'Success']].round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
