"""
config/physics.py

The physical model of the digital twin, in one place.

The environment, the classical baselines and the dashboard all cut with the
same equations, so they import them from here. Before this module each
consumer carried its own copy of the formulas, which is how the wear and
roughness models were able to drift apart in the first place.

Every function takes the coefficient table it should use, so a scenario that
overrides a coefficient gets the physics of that scenario, and a consumer
that uses the calibrated machine gets the calibrated machine. Passing
nothing uses the tables in config/physics_constants.py.

The energy model deserves a note. The specific cutting energy grows with the
cutting speed:

    k_c(v_c) = k_c0 * (v_c / v_ref)^m

so the energy of a finished part is

    E_part = k_c(v_c) * V + P_idle * T_part

The cutting term is fixed by the volume that has to come off, and the idle
term falls when the cycle time falls. Without the speed dependence the two
terms always pointed the same way, the energy objective was a duplicate of
the throughput objective, and the energy weight could not open a trade-off.
With it, the cutting term rises as the spindle speeds up while the idle term
falls, the part energy has an interior minimum, and the agent has to choose
where on that curve to sit. That choice is what the energy weight selects.
"""

import numpy as np

from config.physics_constants import ALL_CONSTANTS


def _table(given, name):
    return given if given is not None else ALL_CONSTANTS[name]


def cutting_speed(D, n):
    """Cutting speed [m/min]."""
    return (np.pi * D * n) / 1000.0


def reference_cutting_speed(D, reference=None):
    """Cutting speed of the reference condition [m/min]."""
    ref = reference or ALL_CONSTANTS['reference_cutting_condition']
    return cutting_speed(D, ref['n'])


def specific_cutting_energy(v_c, D, energy_cfg=None, reference=None):
    """Specific cutting energy [W min/mm3] at a cutting speed.

    The reference condition is the anchor: a pass cut under it draws the
    calibrated k_c0, and the coefficient grows as the spindle speeds up.
    """
    energy = _table(energy_cfg, 'energy')
    exponent = energy.get('k_c_speed_exponent', 0.0)
    if exponent == 0.0:
        return energy['k_c']
    ratio = np.maximum(v_c / reference_cutting_speed(D, reference), 1e-9)
    return energy['k_c'] * ratio ** exponent


def material_removal_rate(v_c, a_p, f_r):
    """Material removal rate [mm3/min]."""
    return v_c * a_p * f_r


def cutting_power(v_c, MRR, D, energy_cfg=None, reference=None):
    """Power drawn by the cutting process [W]."""
    return specific_cutting_energy(v_c, D, energy_cfg, reference) * MRR


def pass_energy(n, f_r, a_p, D, L_cut, energy_cfg=None, reference=None):
    """Energy of one pass [kJ].

    The power is the idle power of the machine plus the power of the cutting
    process, held for the time the pass takes.
    """
    energy = _table(energy_cfg, 'energy')
    v_c = cutting_speed(D, n)
    MRR = material_removal_rate(v_c, a_p, f_r)
    power = energy['P_idle'] + cutting_power(v_c, MRR, D, energy_cfg, reference)
    t_cut = L_cut / (n * f_r)
    return power * t_cut * 60 / 1000.0


def reference_pass_energy(D, L_cut, energy_cfg=None, reference=None):
    """Energy of a pass cut under the reference condition [kJ]."""
    ref = reference or ALL_CONSTANTS['reference_cutting_condition']
    return pass_energy(ref['n'], ref['f_r'], ref['a_p'], D, L_cut,
                       energy_cfg, reference)


def flank_wear_rate(v_c, f_r, a_p, t_cut, wear_cfg=None):
    """Flank wear added by one pass [mm], Taylor-expanded."""
    wear = _table(wear_cfg, 'tool_wear')
    return (wear['C_wear'] * v_c ** wear['alpha'] * f_r ** wear['beta']
            * a_p ** wear['gamma_wear'] * t_cut)


def surface_roughness(f_r, VB, v_c, roughness_cfg=None):
    """Surface roughness of the pass [um]."""
    roughness = _table(roughness_cfg, 'surface_roughness')
    return roughness['k1'] * f_r ** 2 + roughness['k2'] * VB + roughness['k3'] / v_c
