import gymnasium as gym
from gymnasium.envs.registration import register
from config.physics_constants import (
    BATCH_PARAMETERS,
    ENGINEERING_CONSTRAINTS,
    WORKPIECE_PARAMETERS,
    MACHINE_CONSTRAINTS
)

# Register once
register(
    id="MillingDTEnv-v1",
    entry_point="env.milling_env:MillingDTEnv",
    max_episode_steps=200,
    kwargs={
        "kappa_total": WORKPIECE_PARAMETERS['kappa_total'],
        "VB_max": ENGINEERING_CONSTRAINTS['VB_max'],
        "Ra_max": ENGINEERING_CONSTRAINTS['Ra_max'],
        "D": WORKPIECE_PARAMETERS['D'],
        "L_cut": WORKPIECE_PARAMETERS['L_cut'],
        "n_min": MACHINE_CONSTRAINTS['n_min'],
        "n_max": MACHINE_CONSTRAINTS['n_max'],
        "f_min": MACHINE_CONSTRAINTS['f_r_min'],
        "f_max": MACHINE_CONSTRAINTS['f_r_max'],
        "ap_min": MACHINE_CONSTRAINTS['a_p_min'],
        "ap_max": MACHINE_CONSTRAINTS['a_p_max'],
    }
)

from env.milling_env import MillingDTEnv

__all__ = ["MillingDTEnv"]
