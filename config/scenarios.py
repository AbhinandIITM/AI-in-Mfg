"""
config/scenarios.py

Randomized machining scenarios.

The environment samples its training scenarios here and the Monte Carlo sweep
samples its evaluation scenarios from the same tables, so the policy is
trained on the distribution it is measured on and the two cannot drift apart.
"""

import numpy as np

from config.physics_constants import (
    ALL_CONSTANTS,
    SCENARIO_BANDS,
    SCENARIO_INITIAL_WEAR,
    SCENARIO_RANGES,
)

# The physical values a scenario can set. Anything else passed to the
# environment (reward weights, machine limits, energy model) is not part of a
# scenario and is not randomized.
SCENARIO_KEYS = (
    'kappa_total', 'D', 'L_cut', 'VB_0',
    'C_wear', 'alpha', 'beta', 'gamma_wear',
    'k1', 'k2', 'k3',
)

def sample_scenario(rng, exclude=()):
    """Draw one machining scenario.

    Args:
        rng: numpy random generator.
        exclude: keys that must keep their pinned value.
    """
    scenario = {}

    for key, (low, high) in SCENARIO_RANGES.items():
        scenario[key] = float(rng.uniform(low, high))

    wear = ALL_CONSTANTS['tool_wear']
    roughness = ALL_CONSTANTS['surface_roughness']
    for key, (low, high) in SCENARIO_BANDS.items():
        nominal = wear[key] if key == 'C_wear' else roughness[key]
        scenario[key] = float(nominal * rng.uniform(low, high))

    wear_limit = ALL_CONSTANTS['engineering']['VB_max']
    scenario['VB_0'] = float(rng.uniform(*SCENARIO_INITIAL_WEAR) * wear_limit)

    for key in exclude:
        scenario.pop(key, None)

    return scenario

def generate_random_configs(n=100, seed=42):
    """Named scenarios for the Monte Carlo sweep.

    The same set is used for every policy that gets compared, so the rows
    compare policies rather than scenarios.
    """
    rng = np.random.default_rng(seed)
    configs = {}
    for i in range(n):
        configs[f"Scenario_{i:03d}"] = sample_scenario(rng)
    return configs
