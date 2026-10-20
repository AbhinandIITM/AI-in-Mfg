# Environment Specification and Acceptance Status

**Subject:** `env/milling_env.py`, the two registered environments
**Status:** matches the implementation as of the last commit
**Related:** `README.md` (overview), `docs/references.md` (where the physics comes from),
`docs/LITERATURE_COMPARISON.md` (position against the published work)

This document replaces the original project plan, whose file structure and
reward description no longer matched the code.

---

## 1. The Two Registrations

| Registration | Observation | Purpose |
| :--- | :--- | :--- |
| `MillingDTEnv-v1` | 8 features | The policy for a single task. Every physical value can be overridden per scenario. |
| `MillingDTEnv-w2-v1` | 11 features | The same environment, conditioned on the energy weight and the material context. The weight and the scenario are drawn per episode, so one policy covers the whole weight range. |

Both accept the same physical overrides. `MillingDTEnv(randomize=True)` draws a
new machining scenario for every part.

---

## 2. State Space

| # | Feature | Range | Source |
| ---: | :--- | :--- | :--- |
| 0 | Flank wear $VB$ | 0 to $VB_{\max}$, tanh squashed | Taylor-extended wear model |
| 1 | Remaining stock $\kappa$ | 0 to $\kappa_{\text{total}}$, tanh squashed | Progress of the part |
| 2 | Previous spindle speed $n$ | 0 to 1, linear | Last action |
| 3 | Previous feed $f_r$ | 0 to 1, linear | Last action |
| 4 | Previous depth of cut $a_p$ | 0 to 1, linear | Last action |
| 5 | Previous material removal rate | 0 to 1, clipped | $MRR / MRR_{\max}$ |
| 6 | Previous surface roughness $Ra$ | 0 to $Ra_{\max}$, tanh squashed | Empirical regression |
| 7 | Tool broken | 0 or 1 | Breakage event |
| 8 | Energy weight $w_2$ | 0 to 1 | Only in `MillingDTEnv-w2-v1` |
| 9 | Wear coefficient $C_{\text{wear}}$ | 0 to 1 over the scenario band | Only in `MillingDTEnv-w2-v1` |
| 10 | Roughness coefficient $k_1$ | 0 to 1 over the scenario band | Only in `MillingDTEnv-w2-v1` |

The first eight are normalized with the tanh squashing of the plan, which keeps
the gradient magnitude when a constraint is violated.

---

## 3. Action Space

Continuous, three dimensions, each mapped from $[-1, 1]$ onto the machine
envelope and clipped before use:

| Variable | Symbol | Range | Unit |
| :--- | :--- | :--- | :--- |
| Spindle speed | $n$ | 2000 to 8000 | RPM |
| Feed per revolution | $f_r$ | 0.05 to 0.35 | mm/r |
| Depth of cut | $a_p$ | 0.1 to 2.5 | mm |

The depth of cut is additionally capped at the stock that is left, so a pass
never cuts deeper than the remaining allowance.

---

## 4. Transition Equations

$$
v_c = \frac{\pi D n}{1000}, \qquad t_{\text{cut}} = \frac{L_{\text{cut}}}{n f_r}, \qquad MRR = v_c a_p f_r
$$

$$
\Delta VB = C_{\text{wear}} \, v_c^{\alpha} f_r^{\beta} a_p^{\gamma_{\text{wear}}} t_{\text{cut}}, \qquad
Ra = k_1 f_r^2 + k_2 VB + \frac{k_3}{v_c}
$$

$$
E = \left(P_{\text{idle}} + k_c \, MRR\right) t_{\text{cut}}
$$

$E$ is the energy of one pass in kJ. The part energy is the sum over the
passes, which is $k_c V + P_{\text{idle}} T_{\text{part}}$: the cutting term
is fixed by the volume, and the idle term falls when the cycle time falls.

---

## 5. Reward

| Term | Value | Purpose |
| :--- | :--- | :--- |
| Throughput | $+5 \, w_1 \, MRR / MRR_{\max}$ | The objective |
| Energy | $-5 \, w_2 \, E / E_{\text{ref}}$ | The objective |
| Progress | $+2 \, \Delta(\text{stock removed})$ | Anti-stall shaping |
| Flank wear | $-50$ per pass above $VB_{\text{safe}}$ | Guard rail |
| Breakage | $-100$ | Guard rail |
| Roughness | $-50$ if the finishing pass ends above $Ra_{\text{safe}}$ | Guard rail |
| Step | $-0.05$ per pass | Discourage pointless passes |
| Completion | $+200$ when the stock is removed in spec | Discourage stalling |
| Pareto margin | up to $+5$ for the margin below the roughness limit and the reference cycle time | Tie breaker |

$E_{\text{ref}}$ is the energy of a pass cut under the reference condition
(5000 RPM, 0.2 mm/r, 2.5 mm), so a pass under that condition normalizes to 1
and the energy term has the same sensitivity as the throughput term. The
objective terms are multiplied by 5 so that they can compete with the shaping
terms; without that scale the shaping decides the policy and the energy weight
has no effect at all.

$w_1 + w_2 = 1$ in every run.

---

## 6. Episode Logic

- The episode ends when the stock is removed (terminated), when the tool breaks
  (terminated), when the flank wear passes $VB_{\max}$ (terminated), or after
  200 steps (truncated).
- A part starts with a uniformly drawn initial flank wear, or with the pinned
  $VB_0$ of the scenario.
- The info payload carries `pass`, `kappa_rem`, `a_p`, `f_r`, `n`, `Ra`, `MRR`,
  `VB`, `Energy`, `T_total`, `E_total`, `broken` and `is_finishing`.

---

## 7. Constants

All of them live in `config/physics_constants.py`, and the environment, the
baselines and the ablation study read them from there, so no consumer can
drift from another.

| Group | Values |
| :--- | :--- |
| Tool wear | $C_{\text{wear}} = 1.1\times10^{-4}$, $\alpha = 0.3$, $\beta = 0.2$, $\gamma_{\text{wear}} = 0.15$ |
| Surface roughness | $k_1 = 12.5$, $k_2 = 0.8$, $k_3 = 0.5$ |
| Energy | $P_{\text{idle}} = 500$ W, $k_c = 2.5$ W$\cdot$min/mm$^3$ |
| Machine | 2000-8000 RPM, 0.05-0.35 mm/r, 0.1-2.5 mm |
| Engineering | $VB_{\max} = 0.3$ mm, $Ra_{\max} = 1.6$ $\mu$m, safety factor 0.85 |
| Workpiece | $D = 50$ mm, $L_{\text{cut}} = 100$ mm, $\kappa = 5$ mm |

---

## 8. Baselines

| Baseline | Description |
| :--- | :--- |
| Constant high | Maximum speed and feed throughout |
| Constant safe | Minimum speed and feed throughout |
| Rule based | Feed held at maximum, speed dropped as the tool wears |
| NSGA-II static | One parameter set for the whole part, solved with pymoo |
| Receding horizon | Re-solves the next pass at every step, with the remaining stock and the current wear as the state |
| PPO | The learned policy, and the energy weight ablation across seven weights |

The receding-horizon controller is the only classical policy that stays inside
the roughness safety limit, and it reaches a total reward of +204.3 against
-45 for the four static ones. The learned policy and the controller are level
on the reward (204.9 against 204.3), and the controller is ahead on the
material removal rate, the energy and the cycle time. The comparison in
`results/baseline_results_v5.csv` is the one to read.

---

## 9. Acceptance Criteria

| Criterion | Status | Evidence |
| :--- | :--- | :--- |
| Passes `gymnasium.check_env()` with no warnings | Met | `tests/test_env.py::test_gymnasium_env_checker_passes` |
| Two runs with the same seed produce identical results | Met | `tests/test_training.py::test_evaluation_reproducibility`; the Monte Carlo sweep acts deterministically and returns the same numbers on every run |
| Convergence without divergence or NaN | Met | TensorBoard runs under `results/tensorboard/` |
| The Monte Carlo sweep reports 95% confidence intervals for every metric | Met | `results/monte_carlo/eval_PPO_MC_v5.csv`, five metrics with intervals |
| The environment and the baselines use identical constants | Met | `tests/test_env.py::test_physics_constants_sync`; both import `config/physics_constants.py` |
| No duplicated plotting code | Met | `viz/plotting_utils.py` is the single source |
| The ablation demonstrates the energy objective | Met, with a stated limit | The part energy falls by 21% and the cycle time by 53% across the weight grid. The energy of a part and the removal rate are tied together in this energy model, so the weight is not a trade-off against throughput. `docs/LITERATURE_COMPARISON.md` states what would open one. |
