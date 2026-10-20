"""
config

The configuration of the project, in two YAML files:

    physics_constants.yaml   the calibrated physical constants, the reward
                             weights, the reference cutting condition and the
                             scenario tables
    paths.yaml               where the project keeps its runs, models, logs,
                             plots and evaluation tables

This module loads both and exposes them, so the rest of the project imports
plain names and never parses YAML itself.

The equations that consume the constants live in config/physics.py, and the
scenario sampler in config/scenarios.py.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent.parent


def _load(name):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


_PHYSICS = _load("physics_constants.yaml")
_PATHS = _load("paths.yaml")

# ---------------------------------------------------------------- constants
TOOL_WEAR_CONSTANTS = _PHYSICS['tool_wear']
SURFACE_ROUGHNESS_CONSTANTS = _PHYSICS['surface_roughness']
ENERGY_CONSTANTS = _PHYSICS['energy']
MACHINE_CONSTRAINTS = _PHYSICS['machine']
ENGINEERING_CONSTRAINTS = _PHYSICS['engineering']
WORKPIECE_PARAMETERS = _PHYSICS['workpiece']
BATCH_PARAMETERS = _PHYSICS['batch']
REWARD_WEIGHTS = _PHYSICS['reward']
REFERENCE_CUTTING_CONDITION = _PHYSICS['reference_cutting_condition']

# The scenario tables: the ranges of the geometry, the multiplicative bands
# around the nominal coefficients, and the initial wear.
SCENARIO_RANGES = _PHYSICS['scenarios']['ranges']
SCENARIO_BANDS = _PHYSICS['scenarios']['bands']
SCENARIO_INITIAL_WEAR = _PHYSICS['scenarios']['initial_wear']

ALL_CONSTANTS = {
    'tool_wear': TOOL_WEAR_CONSTANTS,
    'surface_roughness': SURFACE_ROUGHNESS_CONSTANTS,
    'energy': ENERGY_CONSTANTS,
    'machine': MACHINE_CONSTRAINTS,
    'engineering': ENGINEERING_CONSTRAINTS,
    'workpiece': WORKPIECE_PARAMETERS,
    'batch': BATCH_PARAMETERS,
    'reward': REWARD_WEIGHTS,
    'reference_cutting_condition': REFERENCE_CUTTING_CONDITION,
    'scenarios': {
        'ranges': SCENARIO_RANGES,
        'bands': SCENARIO_BANDS,
        'initial_wear': SCENARIO_INITIAL_WEAR,
    },
}

# -------------------------------------------------------------------- paths
# Every run of the project lives in its own directory under results/, named
# after the configuration it was trained with. The directory holds the
# policy, the training log, the metrics and the figures of that run, so a
# run can be copied, archived or deleted as one piece.
RESULTS_DIR = PROJECT_ROOT / _PATHS['results']
EVALUATION_DIR = RESULTS_DIR / _PATHS['evaluation']
PLOTS_DIR = RESULTS_DIR / _PATHS['plots']
WORKERS_DIR = RESULTS_DIR / _PATHS['workers']

RUN_PREFIX = _PATHS['run_prefix']

PRIMARY_RUN = _PATHS['primary_run']
CONTROLLER_INITIALIZED_RUN = _PATHS['controller_initialized_run']
TRANSFER_SOURCE_RUN = _PATHS['transfer_source_run']
TRANSFER_SCRATCH_RUN = _PATHS['transfer_scratch_run']


def run_dir(run_name):
    """The directory of one run."""
    return RESULTS_DIR / f"{RUN_PREFIX}{run_name}"


def run_models(run_name):
    """The policy and the normalization statistics of one run."""
    return run_dir(run_name) / 'models'


def run_log(run_name):
    """The training curve of one run."""
    return run_dir(run_name) / 'log'


def run_plots(run_name):
    """The figures of one run."""
    return run_dir(run_name) / 'plots'


def run_metrics(run_name):
    """The evaluation tables of one run."""
    return run_dir(run_name) / 'metrics'


# The policy this project ships with.
PRIMARY_MODEL_DIR = run_models(PRIMARY_RUN)
