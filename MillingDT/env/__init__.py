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

# The batch environment. One episode is a batch of parts cut with one tool,
# so the tool life is the resource the parts share.
#
# The base kwargs are not pinned here: the batch draws the scenario of every
# part at every reset, and pinning the geometry would take the geometry out
# of the draw. The machine limits and the engineering limits come from the
# same configuration the per-part environment reads.
#
# The episode ends when the last part of the batch is cut, which the
# environment decides itself, and the pass limit of the per-part environment
# bounds the passes of every part, so there is no episode limit to set.
BATCH_ENV_ID = "MillingDTBatchEnv-v1"

register(
    id=BATCH_ENV_ID,
    entry_point="MillingDT.env.milling_batch_env:MillingDTBatchEnv",
    # The batch draws a new scenario for the parts at every reset, so the
    # episode is only reproducible when it is seeded.
    nondeterministic=True,
    max_episode_steps=None,
)

from MillingDT.env.milling_env import MillingDTEnv
from MillingDT.env.milling_batch_env import MillingDTBatchEnv

__all__ = ["MillingDTEnv", "MillingDTBatchEnv", "ENV_ID", "BATCH_ENV_ID"]
