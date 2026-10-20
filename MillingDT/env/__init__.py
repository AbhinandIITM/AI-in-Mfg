from gymnasium.envs.registration import register
from MillingDT.config import (
    ENGINEERING_CONSTRAINTS,
    WORKPIECE_PARAMETERS,
    MACHINE_CONSTRAINTS
)

# The one registered environment. The energy weight and the material context
# are part of the observation, so a single policy covers the whole weight range
# and the whole material distribution; a task with a pinned weight is the same
# environment with a constant in the context.
ENV_ID = "MillingDTEnv-v1"

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

register(
    id=ENV_ID,
    entry_point="MillingDT.env.milling_env:MillingDTEnv",
    max_episode_steps=200,
    kwargs=dict(BASE_KWARGS)
)

from MillingDT.env.milling_env import MillingDTEnv

__all__ = ["MillingDTEnv", "ENV_ID"]
