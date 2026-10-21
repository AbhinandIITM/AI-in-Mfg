"""
experiments/reward_sensitivity.py

How much of the comparison between the learned policy and the classical
controller is decided by the reward rather than by the two methods?

The reward credits the removal rate of every pass, which is the standard
way to express a rate in a reward that is paid per pass. It has a known
consequence: the total credit also grows with the number of passes a plan
takes, so a plan that cuts the same part in more, shallower passes is paid
more for it. That is a property of the reward, not of either method, and it
has to be measured rather than argued about.

The script rolls the two arms out on the same 100 randomized scenarios and
accumulates the reward terms in a form that lets the accounting be varied
afterwards:

    the objective as the environment credits it, the sum of the per-pass
    removal rates,
    the same term credited the way it would have to be for the throughput to
    be a part level quantity, the material of the part against the energy of
    the part, which is the alternative that removes the pass count from the
    accounting and makes the throughput weight inert, because the material a
    part gives up is fixed by its stock,
    and the whole reward under a range of objective scales.

The artifact is:
    results/evaluation/reward_sensitivity.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MillingDT.agent.controller_init import controller_action
from MillingDT.config import ALL_CONSTANTS, EVALUATION_DIR, run_models, PRIMARY_RUN
from MillingDT.config.scenarios import generate_random_configs

WEIGHT = 0.4
SCENARIOS = 100
SCALES = [5, 20, 50, 100, 200, 400, 1000, 4000]

REW = ALL_CONSTANTS['reward']


def accumulate(env, step_fn, seed):
    """One episode, with the reward terms in a form that can be recombined.

    The step function performs the step and returns its payload, so the
    environment is advanced exactly once per pass.
    """
    env.reset(seed=seed)
    done = False
    last_progress = 0.0
    objective_sum = 0.0      # the sum of the per-pass rates, the old accounting
    objective_timed = 0.0     # the rates weighted by the duration of the pass
    objective_part = 0.0      # the part level removal against the part level energy
    passes = 0
    penalties = 0.0
    completion = 0.0
    margin = 0.0

    w1 = env.reward_cfg['w1']
    w2 = env.reward_cfg['w2']
    T_ref = env.T_ref

    # The reference part, the part cut under the reference cutting condition.
    # The environment does not expose these, because the reward it pays does
    # not use them, so they are computed here with the same formulas.
    ref = ALL_CONSTANTS['reference_cutting_condition']
    ref_passes = int(np.ceil(env.kappa_total / ref['a_p']))
    ref_pass_time = env.L_cut / (ref['n'] * ref['f_r']) * 60.0
    E_ref_part = ref_passes * env.E_ref
    v_c_ref = (np.pi * env.D * ref['n']) / 1000.0
    MRR_ref = v_c_ref * ref['a_p'] * ref['f_r']
    V_ref = MRR_ref * ref_passes * ref_pass_time / 60.0

    while not done:
        _, _, terminated, truncated, info = step_fn(env)
        done = terminated or truncated
        passes += 1

        # The duration of this pass, in seconds.
        dt_pass = env.L_cut / (info['n'] * info['f_r']) * 60.0

        objective_sum += (w1 * info['MRR'] / env.MRR_max
                          - w2 * info['Energy'] / env.E_ref)
        objective_timed += (w1 * info['MRR'] / env.MRR_max * (dt_pass / T_ref)
                            - w2 * info['Energy'] / env.E_ref)
        # The objective as the part level alternative: the material of the
        # part against the energy of the part.
        objective_part += (w1 * (info['MRR'] * dt_pass / 60.0 / V_ref)
                           - w2 * (info['Energy'] / E_ref_part))

        progress = (env.kappa_total - info['kappa_rem']) / env.kappa_total
        penalties += REW['progress_scale'] * (progress - last_progress)
        last_progress = progress

        if info['VB'] > env.VB_safe:
            penalties -= REW['wear_penalty']
        if info.get('broken'):
            penalties -= REW['breakage_penalty']
        if info.get('is_finishing') and info['Ra'] > env.Ra_safe:
            penalties -= REW['roughness_penalty']
        penalties -= REW['step_penalty']

        if (info.get('is_finishing') and not info.get('broken')
                and info['Ra'] <= env.Ra_safe):
            completion += REW['completion_bonus']
            T_term = max(0.0, 1.0 - info['T_total'] / T_ref)
            Ra_term = max(0.0, 1.0 - info['Ra'] / env.Ra_safe)
            margin += T_term * Ra_term * REW['hv_scale']

    return {
        'objective_sum': objective_sum,
        'objective_timed': objective_timed,
        'objective_part': objective_part,
        'penalties': penalties,
        'completion': completion,
        'margin': margin,
        'passes': passes,
    }


def controller_rollout(scenarios):
    """The controller on the scenarios."""
    rows = []
    for idx, (name, scenario) in enumerate(scenarios.items()):
        env = gym.make("MillingDTEnv-v1", w2=WEIGHT, **scenario).unwrapped

        def step(e):
            action = controller_action(e)
            _, reward, terminated, truncated, info = e.step(action)
            return action, reward, terminated, truncated, info

        rows.append(accumulate(env, step, 42 + idx))
        env.close()
    return rows


def policy_rollout(scenarios, model, vec_norm_path):
    """The learned policy on the same scenarios."""
    rows = []
    for idx, (name, scenario) in enumerate(scenarios.items()):
        wrapped = gym.make("MillingDTEnv-v1", w2=WEIGHT, **scenario)
        env = wrapped.unwrapped
        vec = DummyVecEnv([lambda: wrapped])
        vec = VecNormalize.load(str(vec_norm_path), vec)
        vec.training = False
        vec.norm_reward = False
        state = {"obs": None}

        def step(e, vec=vec, state=state):
            if state["obs"] is None:
                state["obs"] = vec.reset()
            action, _ = model.predict(state["obs"], deterministic=True)
            state["obs"], reward, done, info = vec.step(action)
            return action, float(reward[0]), bool(done[0]), bool(done[0]), info[0]

        rows.append(accumulate(env, step, 42 + idx))
        vec.close()
    return rows


def main():
    scenarios = generate_random_configs(SCENARIOS, seed=42)
    model_dir = run_models(PRIMARY_RUN)
    model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")

    arms = {
        "Receding-horizon controller": controller_rollout(scenarios),
        "PPO": policy_rollout(scenarios, model, model_dir / "vecnormalize.pkl"),
    }

    records = []
    for arm, rows in arms.items():
        df = pd.DataFrame(rows)
        for scale in SCALES:
            for form, column in (("the sum of the per-pass rates", "objective_sum"),
                                 ("the material of the part against the energy of the part",
                                  "objective_part")):
                total = (scale * df[column] + df['penalties']
                         + df['completion'] + df['margin']).mean()
                records.append({
                    'arm': arm,
                    'objective_form': form,
                    'objective_scale': scale,
                    'mean_reward': total,
                    'passes_per_part': df['passes'].mean(),
                    'objective_term': df[column].mean(),
                })

    summary = pd.DataFrame(records)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "reward_sensitivity.csv"
    summary.to_csv(out, index=False)

    print("Reward sensitivity, 100 randomized scenarios at w2 = 0.4\n")
    for form in ("the sum of the per-pass rates",
                 "the material of the part against the energy of the part"):
        print(f"--- the objective term, {form} ---")
        pivot = summary[summary['objective_form'] == form].pivot(
            index='objective_scale', columns='arm', values='mean_reward')
        pivot['difference'] = pivot['PPO'] - pivot['Receding-horizon controller']
        print(pivot.round(2).to_string())
        print()

    passes = {arm: pd.DataFrame(rows)['passes'].mean() for arm, rows in arms.items()}
    print("passes per part: " + ", ".join(f"{arm} {value:.2f}" for arm, value in passes.items()))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
