"""
MillingDT

The digital twin of a CNC milling machine and the policies that drive it.

The package holds everything that is reusable:

    config/      the calibrated constants and the paths, in YAML
    env/         the one registered Gymnasium environment
    agent/       the training, the evaluation and the experiment scripts
    baselines/   the classical policies and the receding-horizon controller
    sim/         the pass-by-pass simulator behind the dashboard
    utils/       the seeding
    viz/         the figures

The runs themselves live outside the package, under experiments/, one
directory per training configuration.
"""

from MillingDT.env import ENV_ID, MillingDTEnv
from MillingDT.config import ALL_CONSTANTS

__all__ = ["ENV_ID", "MillingDTEnv", "ALL_CONSTANTS"]
