"""
baselines/run_baselines.py
Implementation of 4 non-RL baselines for MillingDTEnv:
1. Constant-High
2. Constant-Safe
3. Rule-Based Degradation
4. NSGA-II Static (pymoo)

Outputs:
    results/baseline_results_v5.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env # Register environment
from config.physics_constants import ALL_CONSTANTS
import gymnasium as gym

# ---------------------------------------------------------------------------
# NSGA-II Problem Definition
# ---------------------------------------------------------------------------
class MillingStaticProblem(Problem):
    """
    Static Multi-Objective Optimization Problem for NSGA-II.
    Objective 1: Maximize MRR (minimize -MRR)
    Objective 2: Minimize Energy
    Constraints: Ra <= Ra_max, VB <= VB_max after N parts.
    Decision variables: [n, f_r, a_p] fixed for the entire batch.
    """
    def __init__(self):
        machine = ALL_CONSTANTS['machine']
        eng = ALL_CONSTANTS['engineering']
        wp = ALL_CONSTANTS['workpiece']

        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([machine['n_min'], machine['f_r_min'], machine.get('a_p_min', 0.1)]),
            xu=np.array([machine['n_max'], machine['f_r_max'], machine.get('a_p_max', 2.5)])
        )
        self.constants = ALL_CONSTANTS

    def _evaluate(self, x, out, *args, **kwargs):
        n_vals = x[:, 0]
        f_vals = x[:, 1]
        ap_vals = x[:, 2]

        mrr_out = np.zeros(len(x))
        e_out = np.zeros(len(x))
        ra_viol = np.zeros(len(x))
        vb_viol = np.zeros(len(x))

        wear_cfg = self.constants['tool_wear']
        rough_cfg = self.constants['surface_roughness']
        energy_cfg = self.constants['energy']
        eng_cfg = self.constants['engineering']
        wp_cfg = self.constants['workpiece']

        for i in range(len(x)):
            n = n_vals[i]
            f_r = f_vals[i]
            a_p = ap_vals[i]

            # Physics per pass
            v_c = (np.pi * wp_cfg['D'] * n) / 1000.0
            t_cut = wp_cfg['L_cut'] / (n * f_r)
            MRR = v_c * a_p * f_r
            P_cutting = energy_cfg['k_c'] * MRR
            E_pass = (energy_cfg['P_idle'] + P_cutting) * t_cut * 60 / 1000.0

            # Simulate a full batch (kappa_total / a_p passes)
            n_passes = int(np.ceil(wp_cfg['kappa_total'] / a_p))

            VB = 0.0
            total_E = 0.0
            max_Ra = 0.0

            for _ in range(n_passes):
                Ra = rough_cfg['k1'] * (f_r ** 2) + rough_cfg['k2'] * VB + rough_cfg['k3'] / v_c
                if Ra > max_Ra: max_Ra = Ra

                delta_VB = wear_cfg['C_wear'] * (v_c ** wear_cfg['alpha']) * \
                           (f_r ** wear_cfg['beta']) * (a_p ** wear_cfg['gamma_wear']) * t_cut
                VB += delta_VB
                total_E += E_pass

            mrr_out[i] = MRR
            e_out[i] = total_E

            # Constraints (<= 0 is valid in pymoo)
            ra_viol[i] = max_Ra - eng_cfg['Ra_max']
            vb_viol[i] = VB - eng_cfg['VB_max']

        # NSGA-II minimizes objectives. We want to maximize MRR -> minimize -MRR
        out["F"] = np.column_stack([-mrr_out, e_out])
        out["G"] = np.column_stack([ra_viol, vb_viol])

def solve_nsga2() -> np.ndarray:
    """Finds a Pareto-optimal static action and selects a balanced knee point."""
    problem = MillingStaticProblem()
    algorithm = NSGA2(pop_size=100)
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 100),
        seed=1,
        verbose=False
    )

    if res.X is None or len(res.X) == 0:
        print("NSGA-II found no feasible solutions! Falling back to safe.")
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    # Select knee point
    ptp_val = np.ptp(res.F, axis=0)
    F_norm = (res.F - res.F.min(axis=0)) / (ptp_val + 1e-9)
    dist = np.sum(F_norm ** 2, axis=1)
    best_idx = np.argmin(dist)

    best_real_action = res.X[best_idx]

    # Convert real n, f_r, a_p to normalized [-1, 1] action
    m = ALL_CONSTANTS['machine']
    ap_max_val = m.get('a_p_max', 2.5)
    ap_min_val = m.get('a_p_min', 0.1)
    n_norm = 2.0 * (best_real_action[0] - m['n_min']) / (m['n_max'] - m['n_min']) - 1.0
    f_norm = 2.0 * (best_real_action[1] - m['f_r_min']) / (m['f_r_max'] - m['f_r_min']) - 1.0
    ap_norm = 2.0 * (best_real_action[2] - ap_min_val) / (ap_max_val - ap_min_val) - 1.0

    return np.array([n_norm, f_norm, ap_norm], dtype=np.float32)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_baseline(env_id: str, policy_type: str, n_episodes: int = 50):
    records = []
    env = gym.make(env_id)

    static_nsga_action = None
    if policy_type == "nsga2":
        print("  Solving NSGA-II Pareto front...")
        static_nsga_action = solve_nsga2()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False

        ep_reward = 0.0
        ep_steps = 0
        ep_ra_violations = 0
        ep_energy = 0.0
        ep_mrr_sum = 0.0
        ep_n_sum = 0.0
        ep_f_sum = 0.0
        ep_time = 0.0
        ep_broken = False

        # Access constants for violation check if needed, but we use info
        Ra_max = ALL_CONSTANTS['engineering']['Ra_max']

        while not done:
            if policy_type == "constant_high":
                action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            elif policy_type == "constant_safe":
                action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            elif policy_type == "nsga2":
                action = static_nsga_action
            elif policy_type == "rule_based":
                # Max feed, but drop speed by 10% every 2 passes
                drop = (ep_steps // 2) * 0.2
                n_norm = max(1.0 - drop, -1.0)
                action = np.array([n_norm, 1.0, 0.5], dtype=np.float32) # Moderate a_p
            else:
                raise ValueError(f"Unknown policy {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_steps += 1
            if info['Ra'] > Ra_max:
                ep_ra_violations += 1
            ep_energy += info['Energy']
            ep_mrr_sum += info['MRR']
            ep_n_sum += info['n']
            ep_f_sum += info['f_r']
            ep_time = info['T_total']
            if info['broken']:
                ep_broken = True

        records.append({
            "policy": policy_type,
            "episode": ep,
            "total_reward": ep_reward,
            "passes": ep_steps,
            "broken": ep_broken,
            "ra_violation_rate": ep_ra_violations / max(ep_steps, 1),
            "total_energy_kJ": ep_energy,
            "mean_mrr_mm3_min": ep_mrr_sum / max(ep_steps, 1),
            "mean_n_rpm": ep_n_sum / max(ep_steps, 1),
            "mean_f_r_mm_rev": ep_f_sum / max(ep_steps, 1),
            "cycle_time_s": ep_time,
        })

    env.close()
    return records

def evaluate_all_baselines():
    print("=" * 70)
    print("  Evaluating Non-RL Baselines")
    print("=" * 70)

    all_records = []
    baselines = ["constant_high", "constant_safe", "rule_based", "nsga2"]

    for b in baselines:
        print(f"Running {b}...")
        res = run_baseline("MillingDTEnv-v1", b, n_episodes=50)
        all_records.extend(res)

    df = pd.DataFrame(all_records)

    # Save results
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results_v5.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved baseline results to {out_path}")

    # Print summary
    summary = df.groupby("policy").agg({
        "broken": "mean",
        "ra_violation_rate": "mean",
        "total_energy_kJ": "mean",
        "mean_mrr_mm3_min": "mean",
        "cycle_time_s": "mean",
        "total_reward": "mean"
    }).round(4)

    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    evaluate_all_baselines()
