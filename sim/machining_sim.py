"""
sim/machining_sim.py

Physics-based machining simulation engine for CNC milling:
1. Multi-pass material removal and stock depletion calculation.
2. Micro-topography surface roughness profile generation.
3. Controller trajectory simulation (PPO, NSGA-II, Rule-Based, Constant-High).
4. ISO 6983 Fanuc/Siemens compliant G-code block synthesis.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from env.milling_env import MillingDTEnv
from agent.train_rl_ablation import WEIGHTS as ABLATION_WEIGHTS
from config.physics_constants import ALL_CONSTANTS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def load_ppo_policy(model_dir: Path = None):
    """Loads the trained PPO model and VecNormalize statistics."""
    if model_dir is None:
        model_dir = PROJECT_ROOT / "models" / "PPO_seed42"
    
    model_path = model_dir / "final_model.zip"
    vec_path = model_dir / "vecnormalize.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found at {model_path}")
        
    model = PPO.load(str(model_path), device="cpu")
    return model, vec_path

def run_single_simulation(
    policy_type: str = "PPO",
    kappa_total: float = 5.0,
    VB_0: float = 0.02,
    D: float = 50.0,
    L_cut: float = 100.0,
    w1: float = 0.6,
    w2: float = 0.4,
    ppo_model_tuple=None
) -> pd.DataFrame:
    """
    Executes a complete multi-pass machining episode under specified physical conditions.
    Returns a DataFrame with pass-by-pass physical telemetry.
    """
    env_kwargs = {
        "kappa_total": float(kappa_total),
        "D": float(D),
        "L_cut": float(L_cut),
        "VB_0": float(VB_0),
        "w1": float(w1),
        "w2": float(w2),
    }

    test_env = MillingDTEnv(**env_kwargs)

    vec_env = DummyVecEnv([lambda: test_env])
    
    if policy_type == "PPO":
        if ppo_model_tuple is None:
            # Snap the requested energy weight to the closest weight that was
            # actually trained in the ablation study, and load that policy.
            closest_w2 = min(ABLATION_WEIGHTS, key=lambda w: abs(w - w2))
            tag = f"ablation_w2_{closest_w2:.1f}_seed42".replace('.', '_')
            cand_dir = PROJECT_ROOT / "models" / tag
            if (cand_dir / "final_model.zip").exists():
                model, vec_path = load_ppo_policy(cand_dir)
            else:
                model, vec_path = load_ppo_policy()
        else:
            model, vec_path = ppo_model_tuple
            
        vec_env = VecNormalize.load(str(vec_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    obs = vec_env.reset()
    done = False
    pass_records = []
    accum_energy = 0.0
    accum_time = 0.0
    pass_num = 0

    while not done:
        pass_num += 1
        
        if policy_type == "PPO":
            action, _ = model.predict(obs, deterministic=True)
        elif policy_type == "NSGA-II":
            # NSGA-II static Pareto knee parameters (speed ~6200 RPM, feed ~0.15 mm/r, ap max)
            action = np.array([[0.4, -0.33, 1.0]], dtype=np.float32)
        elif policy_type == "Rule-Based":
            # Heuristic feed decay as wear increases
            current_vb = test_env.VB
            wear_ratio = min(current_vb / 0.3, 1.0)
            feed_norm = max(0.8 - 0.6 * wear_ratio, -0.5)
            action = np.array([[0.5, feed_norm, 1.0]], dtype=np.float32)
        elif policy_type == "Constant-High":
            action = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        elif policy_type == "Constant-Safe":
            action = np.array([[-1.0, -1.0, -1.0]], dtype=np.float32)
        else:
            action = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        if action.ndim == 1:
            action = action.reshape(1, -1)

        obs, rew, done, info_list = vec_env.step(action)
        info = info_list[0]
        
        accum_energy += info["Energy"]
        v_c = (np.pi * D * info["n"]) / 1000.0
        t_cut = L_cut / (info["n"] * info["f_r"])
        accum_time += t_cut
        cutting_power = ALL_CONSTANTS["energy"]["P_idle"] + ALL_CONSTANTS["energy"]["k_c"] * info["MRR"]

        pass_records.append({
            "pass": pass_num,
            "policy": policy_type,
            "n_rpm": info["n"],
            "f_r_mm_rev": info["f_r"],
            "a_p_mm": info["a_p"],
            "kappa_rem_mm": info["kappa_rem"],
            "v_c_m_min": v_c,
            "t_cut_s": t_cut * 60.0,
            "MRR_mm3_min": info["MRR"],
            "cutting_power_W": cutting_power,
            "pass_energy_kJ": info["Energy"],
            "cum_energy_kJ": accum_energy,
            "cum_time_s": accum_time * 60.0,
            "VB_mm": info["VB"],
            "Ra_um": info["Ra"],
            "is_finishing": info["is_finishing"],
            "broken": info.get("broken", False),
            "step_reward": rew[0] if isinstance(rew, (list, np.ndarray)) else rew
        })
        
        if done[0] if isinstance(done, (list, np.ndarray)) else done:
            break

    return pd.DataFrame(pass_records)

def generate_surface_micro_profile(
    final_ra: float, 
    feed_rate: float, 
    tool_radius: float = 0.8, 
    eval_length_mm: float = 4.0, 
    num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic 2D surface micro-topography cusps profile:
    y(x) = cusp_height(x mod feed) + noise
    Returns x (in mm) and z (roughness profile in micrometers).
    """
    x = np.linspace(0, eval_length_mm, num_points)
    f = max(feed_rate, 0.05)
    r = max(tool_radius, 0.2)
    
    # Kinematic roughness peak-to-valley theoretical profile
    # z_th = x_rel^2 / (8 * r)
    x_mod = (x % f) - (f / 2.0)
    z_kinematic_um = (x_mod ** 2) / (8.0 * r) * 1000.0 # to um
    
    # Scale to match simulated Ra
    scale = final_ra / (np.mean(np.abs(z_kinematic_um - np.mean(z_kinematic_um))) + 1e-6)
    z_profile = (z_kinematic_um - np.mean(z_kinematic_um)) * scale
    
    # Add high-frequency physical chatter / shear noise
    np.random.seed(42)
    noise = np.random.normal(0, final_ra * 0.12, num_points)
    z_profile = z_profile + noise
    
    return x, z_profile

def generate_cnc_gcode(
    trajectory_df: pd.DataFrame, 
    part_id: str = "PPO-PART-01", 
    program_num: int = 1001
) -> str:
    """
    Synthesizes compliant ISO 6983 (Fanuc Series 0i / Siemens Sinumerik) G-code program
    corresponding to the exact DRL policy trajectory.
    """
    lines = [
        f"%",
        f"O{program_num:04d} ({part_id} - DRL ADAPTIVE MACHINING)",
        f"(GENERATED BY DIGITAL TWIN DEEP RL CONTROLLER - IIT MADRAS)",
        f"(DATE: 2026 - PPO CONTINUOUS CUTTING OPTIMIZER)",
        f"G21 G90 G94 G40 G80 (METRIC, ABSOLUTE, FEED PER MINUTE)",
        f"G28 G91 Z0. (RETURN TO REFERENCE)",
        f"G90",
        f"T01 M06 (D50 CORNER RADIUS END MILL)",
        f"G54 (WORKPIECE OFFSET 1)",
        f"M08 (FLOOD COOLANT ON)",
        f""
    ]
    
    z_current = 0.0
    
    for idx, row in trajectory_df.iterrows():
        p_num = int(row["pass"])
        n_rpm = int(row["n_rpm"])
        f_r = float(row["f_r_mm_rev"])
        a_p = float(row["a_p_mm"])
        f_min = n_rpm * f_r # Table feed in mm/min
        z_current -= a_p
        
        pass_type = "FINISHING PASS" if row["is_finishing"] else f"ROUGHING PASS #{p_num}"
        
        lines.append(f"(-----------------------------------------------------)")
        lines.append(f"( {pass_type}: ap = {a_p:.3f} mm, fr = {f_r:.3f} mm/rev )")
        lines.append(f"( PREDICTED Ra = {row['Ra_um']:.3f} um, VB = {row['VB_mm']:.4f} mm )")
        lines.append(f"(-----------------------------------------------------)")
        lines.append(f"S{n_rpm} M03 (SET SPINDLE SPEED)")
        lines.append(f"G00 X-10.000 Y0.000 (APPROACH POSITION)")
        lines.append(f"G00 Z{z_current + 2.0:.3f} (RAPID TO CLEARANCE)")
        lines.append(f"G01 Z{z_current:.3f} F500.0 (PLUNGE CUT AT AXIAL DEPTH)")
        lines.append(f"G01 X105.000 F{f_min:.1f} (CUTTING FEED ACROSS WORKPIECE)")
        lines.append(f"G00 Z{z_current + 5.0:.3f} (RETRACT CLEAR)")
        lines.append(f"G00 X-10.000 Y0.000")
        lines.append("")
        
    lines.append(f"M09 (COOLANT OFF)")
    lines.append(f"M05 (SPINDLE STOP)")
    lines.append(f"G28 G91 Z0. (ZERO RETURN Z)")
    lines.append(f"G28 G91 Y0. (ZERO RETURN Y - TABLE TO OPERATOR)")
    lines.append(f"M30 (PROGRAM END AND REWIND)")
    lines.append(f"%")
    
    return "\n".join(lines)
