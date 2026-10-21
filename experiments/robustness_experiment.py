"""
experiments/robustness_experiment.py

Where a learned policy can beat a model-based controller.

The controller solves every pass with the exact model and the exact state, so
inside a perfect digital twin it owns the throughput and the energy: it
computes the greedy optimum of the same objective the policy is learning.
There are three regimes where that advantage does not hold, and this script
measures all three on the same scenarios.

1. The nominal machine. The reference comparison.

2. Model error. The machine runs a harder material than the controller was
   built for. The controller solves with the coefficients it believes, the
   policy reads the material it is told about, and the two are compared both
   when the policy is told the truth and when it is not.

3. A noisy measurement of the wear. The controller trusts the wear it is
   given, the policy is trained with the same noise on its observation.

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
    ALL_CONSTANTS,
    EVALUATION_DIR,
    PRIMARY_RUN,
    run_models,
)
from MillingDT.config.scenarios import generate_random_configs
from MillingDT.utils.seeding import set_seed

WEIGHT = 0.4
SCENARIOS = 100

# The measurement noise on the wear and the remaining stock, as a fraction
# of their range. The policy that faces it is trained with the same noise.
NOISE = 0.05
NOISY_RUN = "PPO_noisystate_seed42"

# The machine runs a material that is harder than the controller's model.
# The wear rate is 30% higher and the roughness coefficients are higher, so
# a controller that believes the nominal material pushes the tool past its
# limits.
HARDER_MATERIAL = {
    'C_wear': ALL_CONSTANTS['tool_wear']['C_wear'] * 1.30,
    'k1': ALL_CONSTANTS['surface_roughness']['k1'] * 1.15,
    'k2': ALL_CONSTANTS['surface_roughness']['k2'] * 1.20,
    'k3': ALL_CONSTANTS['surface_roughness']['k3'] * 1.10,
}

# The nominal coefficients, which is what the controller believes.
NOMINAL_MATERIAL = {
    'C_wear': ALL_CONSTANTS['tool_wear']['C_wear'],
    'k1': ALL_CONSTANTS['surface_roughness']['k1'],
    'k2': ALL_CONSTANTS['surface_roughness']['k2'],
    'k3': ALL_CONSTANTS['surface_roughness']['k3'],
}


def evaluate_controller(scenarios, seed=42, weight=WEIGHT, belief=None,
                        material=None, obs_noise=0.0, noise_rng=None):
    """Roll the controller out, with the material and the belief it works with.

    The belief is what the controller thinks the material is. The material
    is what the machine actually cuts, and the physics always run on it.

    With obs_noise the controller is given a noisy reading of the flank wear,
    the same measurement noise the policy sees on its observation.
    """
    rows = []
    for idx, (name, scenario) in enumerate(scenarios.items()):
        env_kwargs = dict(scenario)
        if material:
            env_kwargs.update(material)

        eval_env = gym.make("MillingDTEnv-v1", w2=weight, obs_noise=obs_noise,
                            **env_kwargs)
        eval_env.reset(seed=seed + idx)
        unwrapped = eval_env.unwrapped

        obs, _ = eval_env.reset(seed=seed + idx)
        done = False
        ep_MRR, ep_Energy, ep_Ra, ep_VB = [], [], [], []
        ep_n, ep_f, ep_ap = [], [], []
        info = {}
        total_reward = 0.0
        while not done:
            measured_vb = unwrapped.VB
            if obs_noise and noise_rng is not None:
                measured_vb += noise_rng.normal(0.0, obs_noise * unwrapped.VB_max)
            action = controller_action(unwrapped, belief=belief, vb=measured_vb)
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
                    material=None, belief=None, obs_noise=0.0):
    """The learned policy on the same scenarios, with the same material."""
    env_kwargs = {}
    if material:
        env_kwargs.update(material)
    if belief is not None:
        env_kwargs['belief'] = belief
    if obs_noise:
        env_kwargs['obs_noise'] = obs_noise

    configs = {}
    for name, scenario in scenarios.items():
        configs[name] = {**scenario, **env_kwargs}

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

    # 1. The nominal machine: the controller knows the material it cuts and
    # the policy is told the material it cuts. This is the reference
    # comparison, the one where the controller has every advantage.
    records.append(summarize(
        "Receding-horizon controller", "nominal",
        evaluate_controller(scenarios)))
    records.append(summarize(
        "PPO", "nominal",
        evaluate_policy(model, vec_norm_path, scenarios)))

    # 2. Model error. The machine cuts a harder material than the controller
    # believes. The controller solves with the nominal coefficients, the
    # policy is told the harder material.
    records.append(summarize(
        "Receding-horizon controller", "model error, controller believes nominal",
        evaluate_controller(scenarios, belief=NOMINAL_MATERIAL,
                            material=HARDER_MATERIAL)))
    records.append(summarize(
        "PPO", "model error, policy told the material",
        evaluate_policy(model, vec_norm_path, scenarios,
                        material=HARDER_MATERIAL)))

    # 3. The same error, but the policy is not told either. Now both are
    # wrong about the material, and the question is which one degrades
    # gracefully.
    records.append(summarize(
        "PPO", "model error, policy believes nominal",
        evaluate_policy(model, vec_norm_path, scenarios,
                        material=HARDER_MATERIAL, belief=NOMINAL_MATERIAL)))

    # 4. A noisy measurement of the wear. The controller trusts the reading
    # it gets, the policy was trained with the same noise.
    noisy_dir = run_models(NOISY_RUN)
    if (noisy_dir / "final_model.zip").exists():
        noisy_model = PPO.load(str(noisy_dir / "final_model.zip"), device="cpu")
        records.append(summarize(
            "Receding-horizon controller", "noisy wear measurement",
            evaluate_controller(scenarios, obs_noise=NOISE,
                                noise_rng=np.random.default_rng(7))))
        records.append(summarize(
            "PPO trained with the noise", "noisy wear measurement",
            evaluate_policy(noisy_model, noisy_dir / "vecnormalize.pkl", scenarios,
                            obs_noise=NOISE)))
    else:
        print(f"No policy trained with the noise at {noisy_dir}, skipping that regime.")

    summary = pd.DataFrame(records)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "robustness_results.csv"
    summary.to_csv(out, index=False)

    print("\nRobustness experiment:")
    print(summary[['regime', 'arm', 'mean_reward', 'reward_std', 'MRR_mean',
                   'Energy_per_part', 'Cycle_time', 'Ra_violation_worst_pass',
                   'Broken', 'Success']].round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
