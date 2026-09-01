"""
baselines/run_baselines.py
Implementation of 4 non-RL baselines for MillingDTEnv-v0:
1. Constant-High
2. Constant-Safe
3. Rule-Based Degradation
4. NSGA-II Static (pymoo)

Outputs:
    logs/baseline_results.csv
"""

import sys
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

from env.milling_env import MillingDTEnv

# ---------------------------------------------------------------------------
# NSGA-II Problem Definition
# ---------------------------------------------------------------------------
class MillingStaticProblem(Problem):
    """
    Static Multi-Objective Optimization Problem for NSGA-II.
    Objective 1: Maximize MRR (minimize -MRR)
    Objective 2: Minimize Energy
    Constraints: Ra <= Ra_max, VB <= VB_max after N parts.
    Decision variables: [n, f_r] fixed for the entire batch.
    """
    def __init__(self, env: MillingDTEnv):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([env.n_min, env.f_min]),
            xu=np.array([env.n_max, env.f_max])
        )
        self.env = env

    def _evaluate(self, x, out, *args, **kwargs):
        n_vals = x[:, 0]
        f_vals = x[:, 1]
        
        mrr_out = np.zeros(len(x))
        e_out = np.zeros(len(x))
        ra_viol = np.zeros(len(x))
        vb_viol = np.zeros(len(x))

        for i in range(len(x)):
            n = n_vals[i]
            f_r = f_vals[i]
            
            # Simulate a full batch statically
            VB = 0.0
            total_E = 0.0
            max_Ra = 0.0
            
            for _ in range(self.env.N):
                phys = self.env._compute_physics(n, f_r)
                # Need to compute Ra BEFORE wear update, exactly as environment does
                Ra = self.env.k1 * (f_r ** 2) + self.env.k2 * VB + self.env.k3 / phys["v_c"]
                if Ra > max_Ra: max_Ra = Ra
                VB += phys["delta_VB"]
                total_E += phys["E"]
                
            mrr_out[i] = phys["MRR"]  # MRR is constant
            e_out[i] = total_E
            
            # Constraints (<= 0 is valid in pymoo)
            ra_viol[i] = max_Ra - self.env.Ra_max
            vb_viol[i] = VB - self.env.VB_max

        # NSGA-II minimizes objectives. We want to maximize MRR -> minimize -MRR
        out["F"] = np.column_stack([-mrr_out, e_out])
        out["G"] = np.column_stack([ra_viol, vb_viol])

def solve_nsga2(env: MillingDTEnv) -> np.ndarray:
    """Finds a Pareto-optimal static action using NSGA-II and selects a balanced knee point."""
    problem = MillingStaticProblem(env)
    algorithm = NSGA2(pop_size=100)
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 50),
        seed=1,
        verbose=False
    )
    
    # If no feasible solution found
    if res.X is None or len(res.X) == 0:
        print("NSGA-II found no feasible solutions! Falling back to safe.")
        return np.array([-1.0, -1.0])
        
    # Select the solution closest to the utopia point (knee point)
    # F[:,0] is -MRR (we want to minimize this), F[:,1] is Energy (minimize)
    F_norm = (res.F - res.F.min(axis=0)) / (res.F.ptp(axis=0) + 1e-9)
    dist = np.sum(F_norm ** 2, axis=1)
    best_idx = np.argmin(dist)
    
    best_real_action = res.X[best_idx]
    if best_real_action.ndim > 1:
        best_real_action = best_real_action[0] # handle shape inconsistency
        
    # Convert real n,f to normalized [-1, 1] action for the env
    n_norm = 2.0 * (best_real_action[0] - env.n_min) / (env.n_max - env.n_min) - 1.0
    f_norm = 2.0 * (best_real_action[1] - env.f_min) / (env.f_max - env.f_min) - 1.0
    
    return np.array([n_norm, f_norm], dtype=np.float32)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_baseline(env: MillingDTEnv, policy_type: str, n_episodes: int = 100):
    records = []
    
    # Pre-solve NSGA-II if needed
    static_nsga_action = None
    if policy_type == "nsga2":
        print("  Solving NSGA-II Pareto front...")
        static_nsga_action = solve_nsga2(env)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        
        ep_reward = 0.0
        ep_steps = 0
        ep_ra_violations = 0
        ep_energy = 0.0
        ep_mrr_sum = 0.0
        ep_tool_fail = False

        while not done:
            if policy_type == "constant_high":
                action = np.array([1.0, 1.0], dtype=np.float32)
            elif policy_type == "constant_safe":
                action = np.array([-1.0, -1.0], dtype=np.float32)
            elif policy_type == "nsga2":
                action = static_nsga_action
            elif policy_type == "rule_based":
                # Max feed, but drop speed by 10% (0.2 in normalized space) every 10 parts
                drop = (ep_steps // 10) * 0.2
                n_norm = max(1.0 - drop, -1.0)
                action = np.array([n_norm, 1.0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown policy {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_steps += 1
            ep_ra_violations += int(info.get("Ra_violation", False))
            ep_energy += info.get("E_kJ", 0.0)
            ep_mrr_sum += info.get("MRR_mm3_min", 0.0)
            ep_tool_fail = ep_tool_fail or info.get("tool_failure", False)

        records.append({
            "policy": policy_type,
            "episode": ep,
            "total_reward": ep_reward,
            "batch_complete": ep_steps >= env.N and not ep_tool_fail,
            "tool_failure": ep_tool_fail,
            "ra_violation_rate": ep_ra_violations / max(ep_steps, 1),
            "total_energy_kJ": ep_energy,
            "mean_mrr_mm3_min": ep_mrr_sum / max(ep_steps, 1),
        })

    return records

def evaluate_all_baselines():
    print("=" * 70)
    print("  Evaluating Non-RL Baselines")
    print("=" * 70)
    
    env = MillingDTEnv()
    all_records = []
    
    baselines = ["constant_high", "constant_safe", "rule_based", "nsga2"]
    
    for b in baselines:
        print(f"Running {b}...")
        res = run_baseline(env, b, n_episodes=50)
        all_records.extend(res)
        
    env.close()
    
    df = pd.DataFrame(all_records)
    
    # Save results
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "baseline_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved baseline results to {out_path}")
    
    # Print summary
    summary = df.groupby("policy").agg({
        "batch_complete": "mean",
        "tool_failure": "mean",
        "ra_violation_rate": "mean",
        "total_energy_kJ": "mean",
        "mean_mrr_mm3_min": "mean"
    }).round(4)
    
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    evaluate_all_baselines()
