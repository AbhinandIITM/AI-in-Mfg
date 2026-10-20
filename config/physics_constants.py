"""
Centralized physics and engineering constants.
Based on literature calibration:
  - Tool wear: "DRL-based optimization..." J. Manuf. Processes 156
  - Surface roughness: "Machining parameter optimization..." J. Manuf. Systems 78
"""

# Tool Wear (Taylor-Extended Model)
TOOL_WEAR_CONSTANTS = {
    'C_wear': 1.1e-4,      # Wear coefficient
    'alpha': 0.3,          # Cutting speed exponent
    'beta': 0.2,           # Feed rate exponent
    'gamma_wear': 0.15,    # Depth of cut exponent
}

# Surface Roughness (Empirical Regression)
SURFACE_ROUGHNESS_CONSTANTS = {
    'k1': 12.5,    # Feed rate coefficient
    'k2': 0.8,     # Tool wear coefficient
    'k3': 0.5,     # Cutting speed coefficient
}

# Energy Consumption (Specific Cutting Energy)
ENERGY_CONSTANTS = {
    'P_idle': 500,   # Idle/spindle motor power [W]
    'k_c': 2.5,      # Specific cutting energy at the reference condition [W·min/mm³]
    # Exponent of the speed dependence of the specific cutting energy. The
    # energy per unit volume removed rises as the spindle speeds up, so a
    # faster cut costs more energy per part. Zero restores the speed
    # independent model.
    'k_c_speed_exponent': 0.6,
}

# Machine Constraints
MACHINE_CONSTRAINTS = {
    'n_min': 2000,         # Spindle speed min [RPM]
    'n_max': 8000,         # Spindle speed max [RPM]
    'f_r_min': 0.05,       # Feed rate min [mm/r]
    'f_r_max': 0.35,       # Feed rate max [mm/r]
    'a_p_min': 0.1,        # Depth of cut min [mm]
    'a_p_max': 2.5,        # Depth of cut max [mm]
}

# Engineering Constraints
ENGINEERING_CONSTRAINTS = {
    'VB_max': 0.3,         # Tool wear limit [mm]
    'Ra_max': 1.6,         # Surface roughness limit [um]
    'safety_factor': 0.85, # Safety margin for reward bounds
}

# Workpiece Parameters
WORKPIECE_PARAMETERS = {
    'D': 50,               # Tool/workpiece diameter [mm]
    'L_cut': 100,          # Cutting length [mm]
    'kappa_total': 5.0,    # Total allowance [mm]
}

# Batch Processing
BATCH_PARAMETERS = {
    'batch_size': 50,      # Parts per batch
}

# Reward Weighting
REWARD_WEIGHTS = {
    'w1': 0.6,    # MRR weight (throughput)
    'w2': 0.4,    # Energy weight (green manufacturing — tunable)
}

# Reference cutting condition. A pass cut under these conditions draws the
# reference energy, which is what the reward normalizes the energy term by.
# Normalizing against the extreme instead (slowest feed, deepest cut) leaves
# the energy term an order of magnitude less sensitive than the throughput
# term, and the energy weight stops having a visible effect.
REFERENCE_CUTTING_CONDITION = {
    'n': 5000,        # Spindle speed [RPM]
    'f_r': 0.2,       # Feed per revolution [mm/r]
    'a_p': 2.5,       # Depth of cut [mm]
}

# Ranges of the randomized machining scenarios. The environment samples the
# training scenarios and the Monte Carlo sweep samples the evaluation
# scenarios from these same tables, so the policy is trained on the
# distribution it is measured on.
SCENARIO_RANGES = {
    'kappa_total': (2.0, 15.0),   # Stock allowance [mm]
    'D': (40.0, 150.0),           # Tool/workpiece diameter [mm]
    'L_cut': (50.0, 200.0),       # Cutting length [mm]
}

# Perturbation bands for the coefficients, as a fraction of the calibrated
# value in TOOL_WEAR_CONSTANTS and SURFACE_ROUGHNESS_CONSTANTS.
SCENARIO_BANDS = {
    'C_wear': (0.6, 1.4),
    'k1': (0.8, 1.2),
    'k2': (0.8, 1.2),
    'k3': (0.8, 1.2),
}

# Initial flank wear of the tool, as a fraction of the wear limit VB_max.
SCENARIO_INITIAL_WEAR = (0.0, 0.8)

# Export as dict for easy access
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
}
