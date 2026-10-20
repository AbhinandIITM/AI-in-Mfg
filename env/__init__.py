import gymnasium as gym
from gymnasium.envs.registration import register
from config.physics_constants import (
    BATCH_PARAMETERS,
    ENGINEERING_CONSTRAINTS,
    WORKPIECE_PARAMETERS,
    MACHINE_CONSTRAINTS
)

# Physical defaults of the nominal machining condition. Every registration
# starts from these, and a scenario or a constructor argument overrides them.
BASE_KWARGS = {
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

# Register once
register(
    id="MillingDTEnv-v1",
    entry_point="env.milling_env:MillingDTEnv",
    max_episode_steps=200,
    kwargs=dict(BASE_KWARGS)
)

# Conditioned environment: the observation carries the energy weight and the
# material context, and both the weight and the scenario are drawn per
# episode. One policy trained on this registration covers the whole weight
# range and the whole scenario distribution.
register(
    id="MillingDTEnv-w2-v1",
    entry_point="env.milling_env:MillingDTEnv",
    max_episode_steps=200,
    kwargs=dict(
        BASE_KWARGS,
        condition=True,
        randomize_weight=True,
    )
)

from env.milling_env import MillingDTEnv

__all__ = ["MillingDTEnv"]
