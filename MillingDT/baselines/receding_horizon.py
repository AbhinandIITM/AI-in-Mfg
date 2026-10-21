"""
baselines/receding_horizon.py

Receding-horizon cutting parameter controller.

The static NSGA-II baseline in run_baselines.py picks one parameter set for
the whole part. This controller re-solves at every pass, with the remaining
stock and the current flank wear as the state, which is what a real machine
controller can do.

The solver is a vectorized grid search over the machine envelope. For each
candidate depth of cut it simulates the remaining passes of the part, and it
keeps the candidates that stay inside the limits the reward punishes. The
survivor with the best expected episode reward is the next pass.

The plan of a part is not one parameter set but two: the roughing passes,
all but the last, run on one feed and the finishing pass, the one that
removes the last of the stock, on another. The reward only penalizes the
roughness of the finishing pass, so a plan that holds one feed for the whole
part is forced to satisfy the roughness limit on the worst pass of the part
and throws away the throughput that a roughing feed would have given. A
controller that re-plans at every pass is not obliged to commit to one feed
for the rest of the part, so it plans the pair.

It is a fair baseline because it optimizes the same quantity the environment
pays: the expected episode reward, over a full-part lookahead, with the
model and the state it is given. It is not a stand-in for the reward, and it
does not use knowledge the policy does not have.
"""

import numpy as np

from MillingDT.config import ALL_CONSTANTS
from MillingDT.config.physics import cutting_power, reference_pass_energy


# The reward contract this controller maximizes, in prose.
#
# One pass, with the state before the pass and the chosen (n, f_r, a_p):
#
#   v_c       = pi * D * n / 1000                                        [m/min]
#   t_cut     = L_cut / (n * f_r)                                       [min]
#   MRR       = v_c * a_p * f_r                                         [mm3/min]
#   E_pass    = (P_idle + cutting_power(v_c, MRR, D)) * t_cut * 60 / 1000  [kJ]
#   delta_VB  = C_wear * v_c**alpha * f_r**beta * a_p**gamma_wear * t_cut   [mm]
#   Ra        = k1 * f_r**2 + k2 * VB_before + k3 / v_c                  [um]
#
#   r_objective = OBJECTIVE_SCALE * (w1 * MRR/MRR_max - w2 * E_pass/E_ref)
#   r_progress  = PROGRESS_SCALE * (fraction of the stock this pass removes)
#   r_wear      = -WEAR_PENALTY       if the wear after the pass is above VB_safe
#   r_rough     = -ROUGHNESS_PENALTY  if the finishing pass is above Ra_safe
#   r_step      = -STEP_PENALTY       (every pass)
#   r_tool      = -TOOL_COST * delta_VB
#
# After the pass the tool breaks with the probability
#
#   wear_factor = max(0, (VB_after - 0.15) / (VB_max - 0.15))
#   p_break     = 0.3 * wear_factor * (MRR/MRR_max)**2
#
# A break ends the part at once: the completion bonus is lost and the
# breakage penalty is paid once.
#
# A part finished within spec (not broken, the wear at or below VB_max, the
# finishing pass at or below Ra_safe) earns the completion bonus plus the
# Pareto margin
#
#   HV_SCALE * max(0, 1 - T_total/T_ref) * max(0, 1 - Ra/Ra_safe)
#
# where T_total is the cycle time of the part and T_ref the reference cycle
# time. The controller scores a candidate by the expectation of that sum:
# the per-pass terms are weighted by the probability that the tool survives
# to the pass, the completion bonus and the margin by the probability that
# it survives all of them, and the expected breakage penalty is charged at
# every pass where the tool can break.
#
# Two things follow from the contract, and the controller reads them as the
# reward reads them. The wear penalty lands on every pass that ends above
# VB_safe, so a plan has to keep the wear at or below VB_safe throughout.
# The roughness penalty lands on the finishing pass only, so the feed of the
# roughing passes is free, and only the feed of the last pass has to stay
# inside the roughness limit.


def _candidates(machine_cfg, n_grid=9, f_grid=9):
    """The machine envelope, sampled as a grid of cutting parameters."""
    n_values = np.linspace(machine_cfg['n_min'], machine_cfg['n_max'], n_grid)
    f_values = np.linspace(machine_cfg['f_r_min'], machine_cfg['f_r_max'], f_grid)
    depths = np.linspace(machine_cfg['a_p_min'], machine_cfg['a_p_max'], 6)
    return n_values, f_values, depths


def _passes(kappa_rem, a_p):
    """The roughing passes of a plan, and the stock the finishing one takes.

    Every pass but the last cuts a_p, and the last one takes what is left.
    The number of passes is the same for every candidate at a given depth,
    so the plan is the same shape for all of them.
    """
    n_passes = int(np.ceil(kappa_rem / a_p - 1e-12))
    n_rough = max(n_passes - 1, 0)
    return n_rough, kappa_rem - n_rough * a_p


def _assumptions(VB, kappa_rem, D, L_cut, w1, w2, MRR_max, E_ref,
                 machine_cfg, scenario):
    """The model the controller believes in, and the scales of the reward.

    The scenario holds the coefficients the controller believes the material
    has. They are taken exactly as they are given: the experiment that
    perturbs the twin hands the controller the nominal coefficients on
    purpose, and the controller has no way of knowing better.

    Two quantities of the reward are not part of the state the controller is
    given. The environment scales the progress bonus by the total allowance
    of the part and the Pareto margin by the cycle time of the whole part,
    while the controller is only told the stock that is left. It therefore
    accounts the progress of the remaining part against the remaining stock,
    and it scores the margin against the reference cycle time of the
    remaining part. The progress of a finished part is the same number of
    points for every candidate, so the choice does not depend on it; the
    margin is worth at most HV_SCALE, so the error in it is small next to the
    objectives and the completion bonus.
    """
    constants = ALL_CONSTANTS
    machine_cfg = machine_cfg or constants['machine']

    wear = dict(constants['tool_wear'])
    roughness = dict(constants['surface_roughness'])
    energy = dict(constants['energy'])
    reward = dict(constants['reward'])
    engineering = constants['engineering']

    if scenario:
        for key in ('C_wear', 'alpha', 'beta', 'gamma_wear'):
            if key in scenario:
                wear[key] = float(scenario[key])
        for key in ('k1', 'k2', 'k3'):
            if key in scenario:
                roughness[key] = float(scenario[key])

    if MRR_max is None:
        v_c_max = (np.pi * D * machine_cfg['n_max']) / 1000.0
        MRR_max = v_c_max * machine_cfg['a_p_max'] * machine_cfg['f_r_max']
    if E_ref is None:
        E_ref = reference_pass_energy(D, L_cut, energy)

    safety = engineering.get('safety_factor', 0.85)
    reference = constants['reference_cutting_condition']
    T_ref = ((kappa_rem / reference['a_p'])
             * L_cut / (reference['n'] * reference['f_r']) * 60.0)

    return {
        'VB': float(VB),
        'kappa_rem': float(kappa_rem),
        'D': float(D),
        'L_cut': float(L_cut),
        'w1': float(w1),
        'w2': float(w2),
        'MRR_max': float(MRR_max),
        'E_ref': float(E_ref),
        'T_ref': float(T_ref),
        'machine_cfg': machine_cfg,
        'wear': wear,
        'roughness': roughness,
        'energy': energy,
        'reward': reward,
        'VB_max': float(engineering['VB_max']),
        'VB_safe': float(engineering['VB_max']) * safety,
        'Ra_safe': float(engineering['Ra_max']) * safety,
    }


def _plan_scores(ctx, n_flat, f_rough_flat, f_finish_flat, a_p):
    """The expected episode reward of the part for every candidate.

    A candidate is a spindle speed, a depth of cut and a pair of feeds: the
    roughing passes, all but the last, are cut with f_rough, and the
    finishing pass, the one that removes the last of the stock, with
    f_finish. The depth of cut of the finishing pass is clipped to the stock
    that is left, as the depth of every pass is.

    The value returned is the expectation of the reward the environment pays
    for that part: the terms of a pass are weighted by the probability that
    the tool survives to the pass, the completion bonus and the Pareto
    margin by the probability that it survives all of them, and the breakage
    penalty is charged where the tool can break.

    A break is not drawn here: the survival probability carries it, which is
    the same expectation without the noise of a single realization. The
    simulation stops when the stock is gone and when the wear passes VB_max.

    Returns a dict of arrays, one entry per candidate: the expected reward,
    the feasibility mask, the roughness of the finishing pass, the cycle
    time of the plan and the terms of the first pass.
    """
    reward = ctx['reward']
    wear = ctx['wear']
    roughness = ctx['roughness']
    energy = ctx['energy']

    D = ctx['D']
    L_cut = ctx['L_cut']
    kappa_rem = ctx['kappa_rem']

    count = n_flat.shape[0]
    v_c = (np.pi * D * n_flat) / 1000.0
    r_step = -reward['step_penalty']

    # The roughing block: the same speed, feed and depth on every pass.
    t_cut_rough = L_cut / (n_flat * f_rough_flat)
    MRR_rough = v_c * a_p * f_rough_flat
    E_rough = ((energy['P_idle'] + cutting_power(v_c, MRR_rough, D, energy))
               * t_cut_rough * 60.0 / 1000.0)
    delta_VB_rough = (wear['C_wear'] * v_c ** wear['alpha']
                      * f_rough_flat ** wear['beta'] * a_p ** wear['gamma_wear']
                      * t_cut_rough)
    r_objective_rough = reward['objective_scale'] * (
        ctx['w1'] * MRR_rough / ctx['MRR_max']
        - ctx['w2'] * E_rough / ctx['E_ref'])
    r_progress_rough = reward['progress_scale'] * (a_p / kappa_rem)
    r_tool_rough = -reward['tool_cost'] * delta_VB_rough
    terms_rough = r_objective_rough + r_progress_rough + r_step + r_tool_rough

    # The number of roughing passes and the stock the finishing pass takes.
    n_rough, ap_finish = _passes(kappa_rem, a_p)

    VB_now = np.full(count, ctx['VB'])
    survival = np.ones(count)
    survived = np.zeros(count)
    wear_penalty_rough = np.zeros(count)
    wear_penalty_plain = np.zeros(count)
    wear_ok = np.ones(count, dtype=bool)
    first = None

    for _ in range(n_rough):
        # The pass is executed if the tool survived the passes before it.
        survived += survival
        VB_now = VB_now + delta_VB_rough
        wear_pen = np.where(VB_now > ctx['VB_safe'], -reward['wear_penalty'], 0.0)
        wear_penalty_rough += survival * wear_pen
        wear_penalty_plain += wear_pen
        wear_factor = np.maximum(0.0, (VB_now - 0.15) / (ctx['VB_max'] - 0.15))
        p_break = 0.3 * wear_factor * (MRR_rough / ctx['MRR_max']) ** 2
        if first is None:
            first = {
                'objective': r_objective_rough,
                'progress': np.full(count, r_progress_rough),
                'step': np.full(count, r_step),
                'tool': r_tool_rough,
                'wear': wear_pen,
                # The roughness penalty lands on the finishing pass only.
                'rough': np.zeros(count),
                'p_break': p_break,
                'breakage': survival * p_break * reward['breakage_penalty'],
            }
        survival = survival * (1.0 - p_break)
        wear_ok = wear_ok & (VB_now <= ctx['VB_safe'])

    # The finishing pass: the last of the stock, on its own feed.
    t_cut_finish = L_cut / (n_flat * f_finish_flat)
    MRR_finish = v_c * ap_finish * f_finish_flat
    E_finish = ((energy['P_idle'] + cutting_power(v_c, MRR_finish, D, energy))
                * t_cut_finish * 60.0 / 1000.0)
    delta_VB_finish = (wear['C_wear'] * v_c ** wear['alpha']
                       * f_finish_flat ** wear['beta']
                       * ap_finish ** wear['gamma_wear'] * t_cut_finish)
    Ra_finish = (roughness['k1'] * f_finish_flat ** 2 + roughness['k2'] * VB_now
                 + roughness['k3'] / v_c)
    r_objective_finish = reward['objective_scale'] * (
        ctx['w1'] * MRR_finish / ctx['MRR_max']
        - ctx['w2'] * E_finish / ctx['E_ref'])
    r_progress_finish = reward['progress_scale'] * (ap_finish / kappa_rem)
    r_tool_finish = -reward['tool_cost'] * delta_VB_finish
    r_wear_finish = np.where(VB_now + delta_VB_finish > ctx['VB_safe'],
                              -reward['wear_penalty'], 0.0)
    r_rough_finish = np.where(Ra_finish > ctx['Ra_safe'],
                               -reward['roughness_penalty'], 0.0)
    wear_factor = np.maximum(0.0, (VB_now + delta_VB_finish - 0.15)
                              / (ctx['VB_max'] - 0.15))
    p_break_finish = 0.3 * wear_factor * (MRR_finish / ctx['MRR_max']) ** 2
    terms_finish = (r_objective_finish + r_progress_finish + r_step
                    + r_tool_finish + r_wear_finish + r_rough_finish)

    if first is None:
        # The whole part is one pass, and that pass is the finishing pass.
        first = {
            'objective': r_objective_finish,
            'progress': np.full(count, r_progress_finish),
            'step': np.full(count, r_step),
            'tool': r_tool_finish,
            'wear': r_wear_finish,
            'rough': r_rough_finish,
            'p_break': p_break_finish,
            'breakage': np.full(count, p_break_finish * reward['breakage_penalty']),
        }

    survival_finish = survival * (1.0 - p_break_finish)
    VB_last = VB_now + delta_VB_finish
    total_time = (n_rough * t_cut_rough + t_cut_finish) * 60.0

    # The completion bonus and the margin, for the parts that survive every
    # pass and stay inside the limits the environment checks.
    in_spec = (VB_last <= ctx['VB_max']) & (Ra_finish <= ctx['Ra_safe'])
    margin = (reward['hv_scale']
              * np.maximum(0.0, 1.0 - total_time / ctx['T_ref'])
              * np.maximum(0.0, 1.0 - Ra_finish / ctx['Ra_safe']))

    # The expected reward of the part. The breakage penalty over the roughing
    # block is BREAKAGE_PENALTY * (1 - survival): the tool breaks somewhere
    # in the block with that probability, and the penalty is paid once.
    expected = (terms_rough * survived + wear_penalty_rough
                - (1.0 - survival) * reward['breakage_penalty'])
    expected += survival * (terms_finish
                            - p_break_finish * reward['breakage_penalty'])
    expected += survival_finish * (reward['completion_bonus'] + margin) * in_spec

    # The reward of the part if the tool survives every pass.
    no_break = (terms_rough * n_rough + wear_penalty_plain + terms_finish
                + (reward['completion_bonus'] + margin) * in_spec)

    return {
        'expected': expected,
        # Feasible means what the reward punishes: no pass of the plan above
        # the wear safety limit, and the finishing pass inside the roughness
        # limit.
        'feasible': (wear_ok & (VB_last <= ctx['VB_safe'])
                     & (Ra_finish <= ctx['Ra_safe'])),
        'Ra_finish': Ra_finish,
        'total_time': total_time,
        'survival': survival_finish,
        'no_break': no_break,
        'n_rough': n_rough,
        'progress_total': np.full(count, reward['progress_scale']),
        'first': first,
    }


def _solve_plan(VB, kappa_rem, D, L_cut, w1, w2, MRR_max, E_ref,
                machine_cfg, scenario):
    """The plan the controller picks for the state, and what it expects of it.

    The plan is a spindle speed, a depth of cut and a pair of feeds. It is
    the candidate with the highest expected episode reward among the
    feasible ones, or None when nothing in the envelope is safe.
    """
    ctx = _assumptions(VB, kappa_rem, D, L_cut, w1, w2, MRR_max, E_ref,
                       machine_cfg, scenario)

    n_values, f_values, depths = _candidates(ctx['machine_cfg'])
    n_grid, rough_grid, finish_grid = np.meshgrid(n_values, f_values, f_values,
                                                   indexing='ij')
    n_flat = n_grid.ravel()
    rough_flat = rough_grid.ravel()
    finish_flat = finish_grid.ravel()

    best = None
    best_value = -np.inf

    for depth in depths:
        # The depth of cut cannot be below the stability limit of the
        # machine, and it cannot cut more than what is left of the stock.
        a_p = min(float(depth), ctx['kappa_rem'])
        if a_p < ctx['machine_cfg']['a_p_min']:
            continue

        # The last pass of the plan has to be a pass the controller can
        # actually plan: a depth of cut is never below the stability limit of
        # the machine, so a plan that would leave less than that for its last
        # pass could not be carried out, and the conservative corner would
        # cut it instead.
        _, ap_finish = _passes(ctx['kappa_rem'], a_p)
        if ap_finish < ctx['machine_cfg']['a_p_min']:
            continue

        scores = _plan_scores(ctx, n_flat, rough_flat, finish_flat, a_p)
        values = np.where(scores['feasible'], scores['expected'], -np.inf)
        index = int(np.argmax(values))
        if values[index] > best_value:
            best_value = float(values[index])
            best = {
                'n': float(n_flat[index]),
                'f_rough': float(rough_flat[index]),
                'f_finish': float(finish_flat[index]),
                'a_p': a_p,
                'n_rough': int(scores['n_rough']),
                'expected': best_value,
                'survival': float(scores['survival'][index]),
                'no_break': float(scores['no_break'][index]),
                'Ra_finish': float(scores['Ra_finish'][index]),
                'cycle_time': float(scores['total_time'][index]),
                'first': {key: float(value[index])
                          for key, value in scores['first'].items()},
            }

    return best


def solve_next_pass(VB, kappa_rem, D, L_cut, w1=0.6, w2=0.4,
                    MRR_max=None, E_ref=None, machine_cfg=None, scenario=None):
    """Choose the parameters of the next pass.

    The controller maximizes the expected episode reward that the environment
    pays for the part, over a full-part lookahead: for every candidate on
    the grid of the machine envelope it simulates the remaining passes of the
    part, weights the terms of a pass by the probability that the tool
    survives to the pass, and adds the completion bonus and the Pareto
    margin weighted by the probability that the tool survives all of them,
    with the expected breakage penalty charged at every pass where the tool
    can break.

    The plan of a part is a speed, a depth of cut and a pair of feeds: the
    roughing passes run on one feed, the finishing pass on another. The
    reward only penalizes the roughness of the finishing pass, so the plan
    is free to rough on a feed the finishing pass could not survive.

    The search is over the whole grid: 9 speeds, 9 roughing feeds, 9
    finishing feeds and 6 depths of cut, 4374 candidates. It is vectorized
    over the candidates and the depths are a loop, and a solve takes about a
    millisecond.

    It is a fair baseline for the learned policy because it optimizes the
    same objective the policy is rewarded with, over the same horizon, using
    only the model and the state it is given: the flank wear, the remaining
    stock and the coefficients it believes the material has. The candidates
    the reward punishes are excluded with the very limits the reward uses,
    so the controller is neither credited with a risk the policy is punished
    for nor punished for one the reward ignores.

    When no candidate in the envelope is safe, the controller takes the
    conservative corner: the lowest speed, the lowest feed and the remaining
    stock as the depth of cut.

    Args:
        VB: current flank wear [mm].
        kappa_rem: stock still to remove [mm].
        D: tool/workpiece diameter [mm].
        L_cut: cutting length [mm].
        w1, w2: objective weights, throughput and energy.
        MRR_max, E_ref: normalization scales of the reward.
        machine_cfg: machine envelope, defaults to the calibrated one.
        scenario: per-scenario physical overrides (roughness and wear
            coefficients) the controller believes in, defaults to the
            calibrated ones.

    Returns:
        (n, f_r, a_p) in physical units.
    """
    plan = _solve_plan(VB, kappa_rem, D, L_cut, w1, w2, MRR_max, E_ref,
                       machine_cfg, scenario)

    if plan is None:
        # Nothing in the envelope is safe: take the conservative corner.
        machine_cfg = machine_cfg or ALL_CONSTANTS['machine']
        return (machine_cfg['n_min'], machine_cfg['f_r_min'], float(kappa_rem))

    # The next pass is the first pass of the plan: a roughing pass, unless
    # the plan is a single pass, which is then the finishing pass.
    feed = plan['f_finish'] if plan['n_rough'] == 0 else plan['f_rough']
    return plan['n'], feed, plan['a_p']
