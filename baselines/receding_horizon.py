"""
baselines/receding_horizon.py

Receding-horizon cutting parameter controller.

The static NSGA-II baseline in run_baselines.py picks one parameter set for
the whole part. This controller re-solves at every pass, with the remaining
stock and the current flank wear as the state, which is what a real machine
controller can do.

The solver is a vectorized grid search over the machine envelope. For each
candidate depth of cut it simulates the remaining passes of the part and
keeps the candidates that stay inside the roughness and wear limits. The
survivor with the best objective is the next pass.

The objective is the same scalarization the environment rewards with, so the
controller is a good teacher for the reinforcement learning policy.
"""

import numpy as np

from config.physics_constants import ALL_CONSTANTS


def _candidates(machine_cfg, n_grid=9, f_grid=9):
    """The machine envelope, sampled as a grid of cutting parameters."""
    n_values = np.linspace(machine_cfg['n_min'], machine_cfg['n_max'], n_grid)
    f_values = np.linspace(machine_cfg['f_r_min'], machine_cfg['f_r_max'], f_grid)
    depths = np.linspace(machine_cfg['a_p_min'], machine_cfg['a_p_max'], 6)
    return n_values, f_values, depths


def solve_next_pass(VB, kappa_rem, D, L_cut, w1=0.6, w2=0.4,
                    MRR_max=None, E_ref=None, machine_cfg=None, scenario=None):
    """Choose the parameters of the next pass.

    Args:
        VB: current flank wear [mm].
        kappa_rem: stock still to remove [mm].
        D: tool/workpiece diameter [mm].
        L_cut: cutting length [mm].
        w1, w2: objective weights, throughput and energy.
        MRR_max, E_ref: normalization scales of the reward.
        machine_cfg: machine envelope, defaults to the calibrated one.
        scenario: per-scenario physical overrides (roughness and wear
            coefficients), defaults to the calibrated ones.

    Returns:
        (n, f_r, a_p) in physical units.
    """
    constants = ALL_CONSTANTS
    machine_cfg = machine_cfg or constants['machine']
    wear = dict(constants['tool_wear'])
    roughness = dict(constants['surface_roughness'])
    energy = dict(constants['energy'])
    eng = constants['engineering']

    if scenario:
        for key in ('C_wear', 'alpha', 'beta', 'gamma_wear'):
            if key in scenario:
                wear[key] = float(scenario[key])
        for key in ('k1', 'k2', 'k3'):
            if key in scenario:
                roughness[key] = float(scenario[key])

    n_values, f_values, depths = _candidates(machine_cfg)
    n_grid, f_grid = np.meshgrid(n_values, f_values, indexing='ij')
    n_flat = n_grid.ravel()
    f_flat = f_grid.ravel()

    if MRR_max is None:
        v_c_max = (np.pi * D * machine_cfg['n_max']) / 1000.0
        MRR_max = v_c_max * machine_cfg['a_p_max'] * machine_cfg['f_r_max']
    if E_ref is None:
        ref = constants['reference_cutting_condition']
        E_ref = L_cut * (
            energy['P_idle'] / (ref['n'] * ref['f_r'])
            + energy['k_c'] * (np.pi * D / 1000.0) * ref['a_p']
        ) * 60 / 1000.0

    Ra_safe = eng['Ra_max'] * eng.get('safety_factor', 0.85)
    VB_max = eng['VB_max']

    best = None
    for a_p in depths:
        a_p = min(float(a_p), float(kappa_rem))
        if a_p < machine_cfg['a_p_min']:
            continue
        n_passes = int(np.ceil(kappa_rem / a_p))

        v_c = (np.pi * D * n_flat) / 1000.0
        t_cut = L_cut / (n_flat * f_flat)
        MRR = v_c * a_p * f_flat
        delta_VB = (wear['C_wear'] * v_c ** wear['alpha']
                    * f_flat ** wear['beta'] * a_p ** wear['gamma_wear'] * t_cut)
        energy_pass = (energy['P_idle'] + energy['k_c'] * MRR) * t_cut * 60 / 1000.0

        # Simulate the remaining passes of the part with these parameters.
        VB_now = np.full_like(n_flat, VB)
        max_Ra = np.zeros_like(n_flat)
        for _ in range(n_passes):
            Ra = roughness['k1'] * f_flat ** 2 + roughness['k2'] * VB_now + roughness['k3'] / v_c
            max_Ra = np.maximum(max_Ra, Ra)
            VB_now = VB_now + delta_VB

        feasible = (max_Ra <= Ra_safe) & (VB_now <= VB_max)
        if not feasible.any():
            continue

        score = w1 * (MRR / MRR_max) - w2 * (energy_pass / E_ref)
        score = np.where(feasible, score, -np.inf)
        index = int(np.argmax(score))
        if best is None or score[index] > best[0]:
            best = (float(score[index]), float(n_flat[index]), float(f_flat[index]), a_p)

    if best is None:
        # Nothing in the envelope is safe: take the conservative corner.
        return machine_cfg['n_min'], machine_cfg['f_r_min'], float(kappa_rem)

    _, n_best, f_best, a_best = best
    return n_best, f_best, a_best
