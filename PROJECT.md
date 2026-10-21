# Digital Twin CNC Milling: Project Documentation

Every plan, specification and reference document of this project, in one file.
It is the consolidation of the documents that used to live in `docs/` and at the
root, and nothing in it is written twice.

| Part | What it holds |
| :--- | :--- |
| 1 | The environment contract: state, actions, physics, reward, episode logic, constants, and the acceptance criteria with their measured status |
| 2 | The method upgrade: domain randomization, the policy conditioned on the energy weight, the transfer experiment, the receding-horizon controller, the three fixes that made the comparison against that controller meaningful, the robustness experiment, the regime map that measures the boundary between the two methods, and the batch experiment, with the measured outcome of each |
| 3 | The position against the three papers in `docs/papers/`, with every number either quoted from a paper or measured by a script in this repository |
| 4 | The design of the Streamlit workstation |
| 5 | Where every physics equation comes from |
| 6 | The original improvement plan of October 8, 2026, kept as written |
| 7 | The audit of that plan, and the defects it found |

The entry point for the project is `README.md`. The measured results are in
`results/RESULTS.md`, which is generated from the artifacts by
`python -m experiments.summarize_results`.

## Contents

1. [Environment Specification](#part-1-environment-specification)
2. [Method Upgrade Plan](#part-2-method-upgrade-plan)
3. [Literature Comparison](#part-3-literature-comparison)
4. [Interactive Simulator Plan](#part-4-interactive-simulator-plan)
5. [Physics References](#part-5-physics-references)
6. [Original Improvement Plan](#part-6-original-improvement-plan)
7. [Audit of the Original Plan](#part-7-audit-of-the-original-plan)

---

## Part 1: Environment Specification


---

**Subject:** `MillingDT/env/milling_env.py`, the one registered environment
**Status:** matches the implementation as of the last commit
**Related:** `README.md` (overview), [Part 5, Physics References](#part-5-physics-references) (where the physics comes from),
[Part 3, Literature Comparison](#part-3-literature-comparison) (position against the published work)

This document replaces the original project plan, whose file structure and
reward description no longer matched the code.

---

### 1. The Registration

| Registration | Observation | Purpose |
| :--- | :--- | :--- |
| `MillingDTEnv-v1` | 11 features | The one environment. The energy weight and the material context are in the observation, so a single policy covers the whole weight range and the whole material distribution. A task with a pinned weight is the same environment with a constant in the context. |
| `MillingDTBatchEnv-v1` | 13 features | The batch environment. One episode is a batch of ten parts cut with one tool, so the tool life is the resource the parts share. The two extra features are the fraction of the tool life left and the fraction of the batch still to cut. |

The identifier of the per-part environment is `MillingDT.ENV_ID`, and every
consumer takes it from there:
`experiments/train_rl.py`, `experiments/evaluate.py`,
`experiments/transfer_experiment.py` and `experiments/run_baselines.py` all
build their environments from it. The batch environment has its own
identifier, `MillingDT.env.BATCH_ENV_ID`, and `experiments/batch_experiment.py`
builds its environment from that.

It accepts the same physical overrides per scenario.
`MillingDTEnv(randomize=True)` draws a new machining scenario for every part
and `MillingDTEnv(randomize_weight=True)` draws a new energy weight for every
part.

---

### 2. State Space

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
| 8 | Energy weight $w_2$ | 0 to 1 | The weight this part is optimized for |
| 9 | Wear coefficient $C_{\text{wear}}$ | 0 to 1 over the band $[0.6, 1.4]$ of nominal | The material, as the policy sees it |
| 10 | Roughness coefficient $k_1$ | 0 to 1 over the band $[0.8, 1.2]$ of nominal | The material, as the policy sees it |

The first eight are normalized with the tanh squashing of the plan, which keeps
the gradient magnitude when a constraint is violated. The last three are the
conditioning context, and they are always present: a policy trained with a
pinned weight sees a constant in the first of them, and every policy in the
repository works on the same 11-dimensional observation.

The batch environment adds two features to that observation, the fraction of
the tool life that is left and the fraction of the batch that is still to cut,
so it carries 13 in all. The physics of a part is the physics of the per-part
environment: the batch environment decides when a part is over, moves the tool
into the next part and keeps the count of the tool changes, and it adds
nothing to the equations.

---

### 3. Action Space

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

### 4. Transition Equations

$$
v_c = \frac{\pi D n}{1000}, \qquad t_{\text{cut}} = \frac{L_{\text{cut}}}{n f_r}, \qquad MRR = v_c a_p f_r
$$

$$
\Delta VB = C_{\text{wear}} \, v_c^{\alpha} f_r^{\beta} a_p^{\gamma_{\text{wear}}} t_{\text{cut}}, \qquad
Ra = k_1 f_r^2 + k_2 VB + \frac{k_3}{v_c}
$$

$$
k_c(v_c) = k_{c0} \left(\frac{v_c}{v_{\text{ref}}}\right)^{m}, \qquad
E = \left(P_{\text{idle}} + k_c(v_c) \, MRR\right) t_{\text{cut}}
$$

$E$ is the energy of one pass in kJ. The specific cutting energy grows with
the cutting speed, so the energy of a finished part is

$$
E_{\text{part}} = k_c(v_c) \cdot V + P_{\text{idle}} \cdot T_{\text{part}}
$$

The cutting term rises as the spindle speeds up, because the energy per unit
volume removed grows, and the idle term falls because the cycle time falls.
The part energy therefore has an interior minimum, and the throughput keeps
rising with the speed: the energy objective and the throughput objective pull
in opposite directions, and the energy weight chooses where between them the
policy sits. Without the speed dependence the cutting term was fixed by the
volume, both terms moved the same way, and the energy objective was a
duplicate of the throughput objective.

$m$ is 0.6 and $v_{\text{ref}}$ is the cutting speed of the reference
condition, 785 m/min at 50 mm and 5000 RPM, so the calibrated $k_{c0} = 2.5$
is the coefficient that applies at the reference condition.

---

### 5. Reward

| Term | Value | Purpose |
| :--- | :--- | :--- |
| Throughput | $+100 \, w_1 \, MRR / MRR_{\max}$ | The objective |
| Energy | $-100 \, w_2 \, E / E_{\text{ref}}$ | The objective |
| Progress | $+20$ per unit of the allowance removed | Anti-stall shaping |
| Flank wear | $-50$ per pass above $VB_{\text{safe}}$ | Guard rail |
| Breakage | $-100$ | Guard rail |
| Roughness | $-50$ if the finishing pass ends above $Ra_{\text{safe}}$ | Guard rail |
| Step | $-0.05$ per pass | Discourage pointless passes |
| Completion | $+200$ when the stock is removed in spec | Discourage stalling |
| Pareto margin | up to $+5$ for the margin below the roughness limit and the reference cycle time | Tie breaker |

The objective term is scaled by 100, the progress term is 20 per unit of the
allowance removed, measured as a fraction of the total allowance, and the
three constraint penalties are flat: 50 for a pass
that ends above the wear safety limit, 100 for a broken tool, and 50 at the
finishing pass when the roughness is above the safety limit. The step penalty
is 0.05, the completion bonus is 200, and the Pareto margin bonus is at most
5, the product of the two margins scaled by 5, so it cannot outvote the
throughput and energy objectives.

The scale of the objective term is a design decision and it is the one that
decides the operating point. At 5 the whole throughput and energy trade-off of
a part was worth about 1.2 points of reward against a completion bonus of
200, so the trade-off was a rounding error in the reward and a policy that
finished every part at any cost scored well. At 100 the trade-off is worth
about fifty points per part, a quarter of the value of a part. The completion
bonus is the value of a part, and the objective term is set against it, so the
bonus still dominates the decision to finish while the throughput decides how
the part is cut. Every one of these magnitudes lives in the `reward` section of
`MillingDT/config/physics_constants.yaml`, and the environment, the controller
and the baselines all read them from there.

$E_{\text{ref}}$ is the energy of a pass cut under the reference condition
(5000 RPM, 0.2 mm/r, 2.5 mm), so a pass under that condition normalizes to 1
and the energy term has the same sensitivity as the throughput term. It is
recomputed for every scenario, from that scenario's diameter, cutting length
and energy coefficient table, so the scale follows the part being cut. The
objective terms are multiplied by 100 so that they can compete with the shaping
terms; without that scale the shaping decides the policy and the energy weight
has no effect at all.

$w_1 + w_2 = 1$ in every run, and the weight belongs to the environment: the
environment reads it from its own reward configuration, so the policy and the
reward can never disagree about which end of the trade-off they are on.

#### The reward is not neutral

The reward credits the removal rate of every pass. That is the standard way to
express a rate in a reward that is paid per pass, but the total credit then
grows with the number of passes a plan takes, so a plan that cuts the same
part in more, shallower passes is paid more for it. It is a property of the
reward, not of either method, and it has to be measured rather than argued
about.

Measured on the same 100 randomized scenarios at $w_2 = 0.4$, by
`experiments/reward_sensitivity.py`:

| | The controller | The policy |
| :--- | ---: | ---: |
| Passes per part | 5.85 | 4.00 |
| Objective credit collected | +0.413 | +0.326 |
| Mean reward | 248.9 | 233.6 |

The two plans remove the same stock allowance. The controller takes more
passes, so it collects 27% more objective credit for the same material, and at
the objective scale of the configuration that is about nine of the fifteen
reward points that separate the two arms in the ideal case. The ideal case
comparison is therefore biased against the learned policy, and the defensible
reading of it is that the two methods are comparable, with the policy ahead on
the physical metrics and behind on the tool breakage. The regime map of
`results/evaluation/regime_map.csv` puts the ideal case at the left end of a
boundary: the crossover is at about 2%, whether the wear measurement is
noisier than 2% of its range or the material model is miscalibrated by more
than 2%, and below it the classical controller is the better operator. The
policy has a limit of its own at about 20% to 25%, where the belief leaves
the distribution it was trained on.

Three things were checked:

1. The objective scale is not the lever. The controller wins at every scale
   from 5 to 4000, and the gap only grows with it, from 7.1 points at 5 to
   352 points at 4000. The sign never flips. The table is in
   `results/evaluation/reward_sensitivity.csv`.
2. The part level alternative was implemented and rejected. Crediting the
   material of the part against the energy of the part removes the pass count
   from the accounting, and under it the gap at the current scale is 5.0
   points instead of 15.3, and the policy is ahead at the two largest scales.
   It is not the right fix, because the material a part gives up is fixed by
   its stock: the volume of a part is `pi * D * kappa * L_cut / 1000`,
   independent of every cutting parameter. The throughput weight would then be
   inert, the reward would be an energy objective with a throughput weight
   that does nothing, and the weight sweep would stop measuring what it
   claims to measure.
3. What a correct formulation would need. The removal rate of a part is the
   ratio of a fixed volume to the part's time, and a rate cannot be
   accumulated over a part without becoming a volume. A correct reward has to
   price the cycle time directly, or compare the part's average rate to a
   reference at the end of the part. Both change the meaning of the weights,
   and the second makes the objective sparse and the credit assignment poor.
   Neither was adopted, and the reward as shipped carries the artifact
   instead, documented rather than hidden.

Tool breakage is stochastic. Above a flank wear of 0.15 mm the probability of
breaking on a pass is

$$
p_{\text{break}} = 0.3 \cdot \frac{VB - 0.15}{VB_{\max} - 0.15} \cdot \left(\frac{MRR}{MRR_{\max}}\right)^2
$$

the wear factor times the square of the load factor. The draw comes from the
environment's own generator, which `reset(seed=...)` seeds, so a scenario
replayed with the same seed breaks in the same place.

---

### 6. Episode Logic

- The episode ends when the stock is removed (terminated), when the tool breaks
  (terminated), when the flank wear passes $VB_{\max}$ (terminated), or after
  200 steps (truncated).
- A part starts with a uniformly drawn initial flank wear, or with the pinned
  $VB_0$ of the scenario.
- The info payload carries `pass`, `kappa_rem`, `a_p`, `f_r`, `n`, `Ra`, `MRR`,
  `VB`, `Energy`, `T_total`, `E_total`, `broken` and `is_finishing`.

---

### 7. Constants

All of them live in `MillingDT/config/physics_constants.yaml`, and the
environment, the baselines and the ablation study read them from there, so no
consumer can drift from another.

| Group | Values |
| :--- | :--- |
| Tool wear | $C_{\text{wear}} = 1.0\times10^{-2}$, $\alpha = 0.3$, $\beta = 0.2$, $\gamma_{\text{wear}} = 0.15$ |
| Surface roughness | $k_1 = 12.5$, $k_2 = 0.8$, $k_3 = 0.5$ |
| Energy | $P_{\text{idle}} = 500$ W, $k_{c0} = 2.5$ W$\cdot$min/mm$^3$, speed exponent $m = 0.6$ |
| Machine | 2000-8000 RPM, 0.05-0.35 mm/r, 0.1-2.5 mm |
| Engineering | $VB_{\max} = 0.3$ mm, $Ra_{\max} = 1.6$ $\mu$m, safety factor 0.85 |
| Workpiece | $D = 50$ mm, $L_{\text{cut}} = 100$ mm, $\kappa = 5$ mm |

#### The wear calibration

The wear coefficient is anchored on the tool life, not on a wear equation taken
from a paper. None of the three reference papers in `docs/papers/` carries a
Taylor wear model: Mo et al. (2025) put the wear limit at 300 um for finishing
and change the tool inside a part, and their virtual environment obtains the
wear value from a trained data-driven model rather than from an analytic wear
law. Wang et al. (2025a) estimate the tool deterioration with a multi-task
time-series model, and in Wang et al. (2025b) the tool state is not a decision
variable at all. There was therefore never a published wear equation to
calibrate the coefficient against, and the previous value of
$1.1\times10^{-4}$ was not calibrated against one either: it gave a tool life
of about 870 parts, so the wear constraint did nothing inside a part and the
tool life was irrelevant to a batch.

The coefficient is now anchored on the tool life that Wang et al. (2025a)
report: 40 parts cut with 5 identical tools, about 8 parts per tool.
$C_{\text{wear}} = 1.0\times10^{-2}$ gives about 9 parts per tool at the
reference cutting condition, which is the same order as the published tool
life. The wear constraint then binds inside a part and the tool life is a
resource a batch can run out of, which is what the batch environment needs.

The exponents $\alpha$, $\beta$ and $\gamma_{\text{wear}}$ are the exponents of
the wear rate, not the Taylor tool life exponents, so the tool life varies by
only a factor of a few across the machine envelope. That is a known weakness of
the model: a faster cut barely shortens the tool life. The classical tool life
equation would give (4.0, 0.8, 3.2), and with them the tool life spans six
orders of magnitude over the envelope, which is not a usable envelope either.

The scenarios are drawn from the same file, and the environment and the Monte
Carlo sweep read the same tables, so the policy is trained on the distribution
it is measured on.

| Quantity | Range |
| :--- | :--- |
| Stock allowance $\kappa_{\text{total}}$ | 2 to 15 mm |
| Tool/workpiece diameter $D$ | 40 to 150 mm |
| Cutting length $L_{\text{cut}}$ | 50 to 200 mm |
| Wear coefficient $C_{\text{wear}}$ | $[0.6, 1.4] \times$ nominal |
| Roughness coefficients $k_1$, $k_2$, $k_3$ | $[0.8, 1.2] \times$ nominal |
| Initial flank wear $VB_0$ | $\mathcal{U}[0, 0.8]$ of $VB_{\max}$ |

The wear exponents $\alpha$, $\beta$ and $\gamma_{\text{wear}}$ are not
randomized. The environment draws a scenario at every `reset()` when it is
built with `randomize=True`, and it recomputes `MRR_max`, `E_ref` and
`T_ref` for every scenario it draws.

---

### 8. Baselines

| Baseline | Description |
| :--- | :--- |
| Constant high | Maximum speed and feed throughout |
| Constant safe | Minimum speed and feed throughout |
| Rule based | Feed held at maximum, speed dropped as the tool wears |
| NSGA-II static | One parameter set for the whole part, solved with pymoo |
| Receding horizon | A part-level maximizer of the expected episode reward, re-planned at every pass, with the remaining stock and the current wear as the state |
| PPO | The learned policy, the production one at `results/run_PPO_seed42` |

The receding-horizon controller is a part-level maximizer of the expected
episode reward. At every pass it simulates the remaining passes of the part for
every candidate on a grid of the machine envelope, weights the terms of a pass
by the probability that the tool survives to it, prices the breakage risk at
every pass where the tool can break, and adds the completion bonus and the
Pareto margin weighted by the probability that the tool survives all of them.
It plans a part with a roughing feed and a finishing feed, because the reward
penalizes the roughness of the finishing pass only. It is a fair baseline
because it optimizes the same quantity the environment pays, with the model
and the state it is given.

The five classical policies are measured on the same 50 episodes of the nominal
condition, at the default energy weight $w_2 = 0.4$.

| Policy | Reward | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Final Ra (um) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Receding horizon | 253.5 | 725.5 | 19.0 | 6.7 | 1.25 |
| Constant high | 10.0 | 1099.6 | 17.8 | 4.3 | 1.56 |
| NSGA-II static | 5.1 | 732.9 | 18.8 | 6.4 | 1.56 |
| Rule based | 3.3 | 719.8 | 18.8 | 6.6 | 1.56 |
| Constant safe | -1882.9 | 1.6 | 385.7 | 768.0 | 0.26 |

Three of the four static policies finish outside the roughness safety limit at
1.56 um, and the fourth, the constant-safe policy, finishes inside it at
0.26 um but takes 768 s and 386 kJ per part, which is what its -1882.9
measures. The receding-horizon controller finishes at 1.25 um, inside the
safety limit, and is the strongest baseline in the repository.

The comparison between the learned policy and the controller is not made in
this table. The controller now optimizes the same expected episode reward the
environment pays, so the comparison has to be made on the scenarios the policy
is measured on, and that is the robustness experiment in
[Part 3, Literature Comparison](#part-3-literature-comparison), section 5.7.
The classical rows are in `results/evaluation/baselines.csv`.

---

### 9. Acceptance Criteria

| Criterion | Status | Evidence |
| :--- | :--- | :--- |
| Passes `gymnasium.check_env()` with no warnings | Met | `tests/test_env.py::test_gymnasium_env_checker_passes` |
| Two runs with the same seed produce identical results | Met | `tests/test_training.py::test_evaluation_reproducibility`; the Monte Carlo sweep acts deterministically and returns the same numbers on every run |
| Convergence without divergence or NaN | Met | Training curves in `results/run_<config>/log/progress.csv`, plotted by `viz/plot_results.py` |
| The Monte Carlo sweep reports 95% confidence intervals for every metric | Met | `results/run_PPO_seed42/metrics/monte_carlo.csv`, five metrics with intervals |
| The environment and the baselines use identical constants | Met | `tests/test_env.py::test_physics_constants_sync`; both import `MillingDT/config/physics_constants.yaml` |
| The environment and the baselines use identical equations | Met | `tests/test_env.py::test_environment_uses_the_shared_physics`; the equations live in `MillingDT/config/physics.py` and every consumer imports them |
| No duplicated plotting code | Met | `viz/plotting_utils.py` is the single source |
| The ablation demonstrates the energy objective | Met | Across the non-dominated weights the removal rate falls 41% and the energy of the part falls 17%, and the spindle speed falls with the energy weight, which is the mechanism the model predicts. [Part 3, Literature Comparison](#part-3-literature-comparison) states the mechanism. |
| One policy covers the whole energy weight range | Met | `results/run_PPO_seed42/metrics/weight_sweep.csv` and `results/evaluation/weight_sweep_vs_ablation.csv`; the single policy removes 1.4% to 5.6% more material for 2.9% to 6.1% less energy over the weights 0.0 to 0.6 than the 21 policies trained per weight, and it is behind at the two heaviest weights, by 9.0% and 21.6% |
| The transfer experiment measures the adaptation against a from-scratch policy | Met | `results/evaluation/transfer_results.csv`; four arms on a harder material, and the production policy reaches 190.1 with no adaptation, the best of the four |
| The controller initialization is measured against the controller it starts from | Met | `results/evaluation/controller_init_results.csv`; three arms on the same 100 randomized scenarios, and the initialized policy is worse than both the controller and the from-scratch policy |
| The classical controller is a baseline for the same objective | Met | `MillingDT/baselines/receding_horizon.py`; it maximizes the expected episode reward the environment pays, over a full-part lookahead, with the breakage risk priced and a separate roughing and finishing feed |
| The comparison against the controller is measured where it decides something | Met | `results/evaluation/robustness_results.csv`; three regimes, and the learned policy wins the two in which the controller's information advantage disappears |
| The boundary between the two methods is measured | Met | `results/evaluation/regime_map.csv`; two sweeps, a noisy wear measurement and a miscalibrated material model, and the crossover is at about 2% on both of them, below it the controller is the better operator and above it the learned policy is. The policy has a limit of its own at about 20% to 25%, where the belief leaves the distribution it was trained on |
| The sensitivity of the comparison against the controller to the reward is measured | Met | `results/evaluation/reward_sensitivity.csv`; the reward credits the removal rate of every pass, and about nine of the fifteen points that separate the two arms in the ideal case are that artifact |
| The batch objective is measured against the per-part objective | Met | `results/evaluation/batch_results.csv`; three arms over 30 batches of ten parts, with the tool change costed at 500 |

---

## Part 2: Method Upgrade Plan


---

**Project:** Energy-Aware DRL for Dynamic Cutting Parameter Optimization (ME6324, IIT Madras)
**Date:** October 20, 2026
**Status:** implemented, with the measured outcomes in sections 10 to 13
**Predecessor:** [Part 7, Audit of the Original Plan](#part-7-audit-of-the-original-plan) (defect remediation, completed) and [Part 3, Literature Comparison](#part-3-literature-comparison)

---

### 1. Where the project stands

After the audit remediation the repository is functional and honest. The measured
baseline, from `results/RESULTS.md`:

- The Monte Carlo sweep runs 100 randomized scenarios (geometry, material and tool
  coefficients, initial flank wear) and reports 95% confidence intervals.
- The energy weight ablation trains 7 weights over 3 seeds and shows a clear
  effect of the energy weight on the operating point: across the frontier the
  spindle speed falls from 7859 RPM at $w_2 = 0.5$ to 5382 RPM at $w_2 = 1.0$,
  and the three lowest weights are dominated.
- The literature comparison quotes only numbers that exist in `docs/papers/` or
  in the artifacts of this repository.

Two measured facts drove this plan.

**Fact 1, the energy objective was not a trade-off in this model.** The energy
of a finished part was

$$E_{\text{part}} = k_c \cdot V + P_{\text{idle}} \cdot T_{\text{part}}$$

The cutting term was fixed by the volume that has to come off, and the idle
term fell exactly when the cycle time fell, which is also what raises the
material removal rate. At the time this plan was written the ablation grid
showed the part energy falling by 21% and the cycle time by 53% as the energy
weight went from 0.0 to 1.0, but the removal rate rose as well. The two
objectives pointed the same way, so the weight sweep produced a monotone
improvement rather than a Pareto frontier, and 21 separate training runs
bought a curve that one policy could produce directly.

**Fact 2, the policy is trained on one operating point and evaluated on many.**
The primary policy was trained on the nominal constants and then evaluated on
randomized scenarios. It was not conditioned on the scenario, so it had to
generalize to material and tool coefficients it had never seen, and the sweep
showed roughness violations in the double digits.

Both facts have the same remedy: put the context into the observation and
train across a distribution of contexts.

Fact 1 also needed a change in the physics rather than in the method. The
specific cutting energy now grows with the cutting speed, so the part energy
has an interior minimum and the energy objective pulls against the throughput
objective instead of with it. The equations are in Part 1, the reasoning in
Part 3, and the measured frontier in `results/RESULTS.md`.

---

### 2. Methods considered, and the decision

| Method | Verdict | Reason |
| :--- | :--- | :--- |
| Multi-objective / conditioned RL (MORL style) | **Adopt** | Replaces the 21-run weight sweep with one policy that is conditioned on the energy weight, and gives the whole frontier from a single training run. |
| Domain randomization in the environment | **Adopt** | The policy trained on one operating point and was measured on many. Training across the same distribution it is measured on is the cheapest possible fix for the robustness claim. |
| Transfer learning | **Adopt** | Maps to a real deliverable (a new material or a different spindle limit), and is cheap because the observation space is fixed. |
| Receding-horizon controller as a teacher, behavioural cloning as warm start | **Adopt the controller, measure the cloning** | A receding-horizon solver is a genuinely strong expert, and it gives the classical comparison a fair baseline. Cloning it into the policy was measured on three arms over the same 100 randomized scenarios: under the current reward it does not produce a policy that beats the controller, it produces a worse one. The record and clone machinery is kept and the initialization is dropped (section 11). |
| Continual learning in the classic sense | **Defer** | There is no task sequence yet. It becomes the right frame only if the target becomes non-stationarity, and then the conditioned policy with replay covers it. |
| Representation learning | **Reject for now** | The state is 8 physics-derived features plus 3 conditioning features, so there is nothing to learn a representation of. It becomes relevant only if raw sensor data (acoustic emission) enters the design, which is a different project. |
| Offline RL | **Reject** | No fixed dataset to be offline about. |
| Pure imitation learning | **Reject as the primary method** | Imitation is for when demonstrations are cheap and reinforcement learning is expensive. Here the simulator is the cheap data source and the reward is available, so imitation throws away information. It is used here as a warm start, not as a method. |
| Curiosity / intrinsic motivation | **Defer** | Exploration is not the bottleneck after the reward rebalance. Worth revisiting only if a task proves hard to learn. |

---

### 3. Phase 1: Domain randomization in the environment

#### Design

`MillingDTEnv` already accepts per-scenario overrides for every physical
coefficient. What it does not do is draw them. The environment gains an
opt-in randomized mode:

- `MillingDTEnv(randomize=True)` samples the scenario at every `reset()`:
  workpiece geometry (`kappa_total`, `D`, `L_cut`), the material and tool
  coefficients (`C_wear`, `k1`, `k2`, `k3`), and the initial flank wear
  (`VB_0`). The wear exponents are not randomized.
- The sampling bands are the same bands the Monte Carlo sweep uses, so the
  policy is trained on the distribution it is measured on. The bands live in
  `MillingDT/config/physics_constants.yaml` so the environment and the sweep
  cannot drift apart.
- The derived scales (`MRR_max`, `E_ref`, `T_ref`) depend on the scenario, so
  they are recomputed whenever the scenario changes. The code as it stood
  computed them once in `__init__`; that moved into an `_apply_scenario()`
  method called from `__init__` and from `reset()`.

#### Expected outcome (hypothesis, to be measured)

Roughness violations in the Monte Carlo sweep fall, because the policy sees
scattered `k1` and pre-worn tools during training instead of meeting them for
the first time during evaluation.

#### Files

- `MillingDT/config/physics_constants.yaml`: the scenario ranges, the bands
  and the initial wear, next to the calibrated constants.
- `MillingDT/config/scenarios.py`: the sampling itself, shared with the Monte Carlo sweep.
- `MillingDT/env/milling_env.py`: `_apply_scenario()`, randomized `reset()`, `randomize` flag.
- `experiments/train_rl.py`: the `--randomize` flag.
- `tests/test_env.py`: a test that two resets of a randomized environment draw
  different scenarios, and that a fixed environment does not.

---

### 4. Phase 2: The conditioned policy

#### Design

The observation carries the energy weight and the scenario context alongside
the eight physics features:

$$s_t = [VB, \kappa, n_{\text{prev}}, f_{\text{prev}}, a_{p,\text{prev}}, MRR_{\text{prev}}, Ra_{\text{prev}}, \text{broken}, w_2, \widetilde{C_{\text{wear}}}, \widetilde{k_1}]$$

The last three features are the conditioning variables. A single policy
trained over a sampled weight space and a sampled scenario distribution
answers, at runtime, "give me the most energy-efficient cut that still meets
the roughness limit for this material".

The plan as written called for a second registration with this observation and
kept the 8-dimensional one for the policies already in `results/`. As built,
there is one registration for the part, `MillingDTEnv-v1`, with the
11-dimensional observation, and a task with a pinned weight is the same
environment with a constant in the context. The two registrations would have
shared every equation anyway, and one environment with one observation is what
the production policy, the 21 ablation policies and the dashboard all use. The
batch environment, added later, is the one task that needs a different
observation, and it is section 13.

#### Validation

The conditioned policy is the claim "one policy covers the frontier". The test
is whether its frontier reproduces the 21-policy frontier measured in
`results/evaluation/ablation_summary.csv`. The two are compared in
`results/evaluation/weight_sweep_vs_ablation.csv`, which
`python -m experiments.evaluate --compare-only` writes.

#### Files

- `MillingDT/env/milling_env.py`: observation composition, the context features.
- `MillingDT/env/__init__.py`: the registration of the part environment and of
  the batch environment.
- `experiments/train_rl.py`: the `--randomize-weight` flag for the production
  policy, and `--w2` for a pinned weight.
- `experiments/evaluate.py`: the `--sweep-weight` and `--compare-only` flags.
- `tests/test_env.py`: observation dimension and content.

---

### 5. Phase 3: Transfer learning across material and machine

#### Design

A self-contained experiment, not a new training mode:

1. Train a source policy on the nominal material (source distribution).
2. Build a target task: a different material (shifted `k1`, `k2`, `C_wear`).
   The plan also called for a different machine (a lower spindle ceiling, a
   different feed limit); as built the target shifts the material only, on the
   same machine.
3. Measure four numbers on the target task:
   - the source policy zero-shot,
   - the source policy after fine-tuning for N episodes,
   - a policy trained on the target from scratch for the same episode budget,
   - the production policy zero-shot, which was trained across a distribution
     of materials and therefore has the material in its observation.
4. Report the adaptation curve: target performance against episodes spent.

The comparison of 2 against 3 is the result that matters: it says how much of
the new task is covered by what the policy already learned. The fourth arm is
the answer the plan expected to find: a policy that has the material in its
observation needs no adaptation step at all.

#### Files

- `experiments/transfer_experiment.py`: the experiment, writing
  `results/evaluation/transfer_results.csv` and an adaptation curve.
- `experiments/summarize_results.py`: add the transfer table.

---

### 6. Phase 4: A receding-horizon controller, and the policy initialized from it

The controller was kept and it was rebuilt. The initialization was measured,
and the measurement is recorded here, because it is the reason for the answer.

#### Design

**The expert.** The existing `experiments/run_baselines.py` solves a static
multi-objective problem once for the whole part. A receding-horizon version
re-solves it at every pass with the current state (remaining stock, current
flank wear, current scenario). This is a legitimate controller, and a much
stronger classical baseline than the static one.

As built, it is not a per-pass greedy solver. It is a part-level maximizer of
the expected episode reward: for every candidate on a grid of the machine
envelope it simulates the remaining passes of the part, weights the terms of a
pass by the probability that the tool survives to it, adds the completion bonus
and the Pareto margin weighted by the probability that the tool survives all of
them, and charges the expected breakage penalty at every pass where the tool
can break. The plan of a part carries a roughing feed and a finishing feed,
because the reward penalizes the roughness of the finishing pass only. The
details and the measurement are in section 11.

**Initializing the policy from it.** The controller's behaviour is recorded on
2000 randomized episodes, as (observation, action) pairs. A supervised actor
with the same shape as the PPO actor (two hidden layers of 256 with Tanh) is
fitted on those trajectories, and the fitted weights are copied into a fresh
PPO policy before the reinforcement learning starts. The intent is that the
learning starts from a feasible behaviour instead of from noise. The machinery
is in `MillingDT/agent/controller_init.py`, and the experiment is
`experiments/controller_experiment.py`.

#### The question and the answer

The question is not whether the initialization helps the learning, it is
whether the learned policy ends up better than the expert it started from.
Three arms, all on the same 100 randomized scenarios at the same energy weight
$w_2 = 0.4$:

| Arm | Mean episode reward | Reward std | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Ra violation, finished surface | Flank wear violations | Tool breakage | Parts completed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Receding-horizon controller | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 2% | 1% | 3% | 96% |
| PPO from scratch | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 4% | 0% | 6% | 94% |
| PPO initialized from the controller | 119.8 | 178.7 | 1274.9 | 81.5 | 42.6 | 4% | 9% | 4% | 87% |

The answer is that the initialization is actively harmful, and the reason is
instructive. The cloned actor starts the policy at the controller's operating
point, which is the conservative one: a low feed and a low spindle speed so
that every pass of the part stays inside the roughness safety limit. With the
wear constraint now binding, that conservatism is expensive. The policy then
learns from a bad starting point, slows down further to stay away from the
wear limit, and ends up cutting a part in 42.6 s against the controller's
17.1 s and the from-scratch policy's 11.4 s. It also exceeds the flank wear
limit on 9% of the parts, which the other two arms never do, and it completes
the fewest parts of the three.

An earlier revision of this document, measured under the old reward, found the
initialization harmless: it made the policy more consistent and did not change
the reward. That was the reward not counting the cost of the slow operating
point. With the throughput counted, the initialization is what the measurement
says it is.

So the method to keep from the controller initialization is the record and
clone machinery, not the initialization itself.

#### Files

- `MillingDT/baselines/receding_horizon.py`: the controller.
- `MillingDT/agent/controller_init.py`: the dataset, the cloned actor and the
  copy into the policy.
- `experiments/run_baselines.py`: adds it to the baseline comparison.
- `experiments/controller_experiment.py`: the three-arm experiment, writing
  `results/evaluation/controller_init_results.csv`.
- `tests/`: the controller produces feasible actions on the nominal scenario.

---

### 7. Out of scope

- Representation learning, offline RL, curiosity, and classic continual
  learning, for the reasons in section 2. They are listed so that the
  decision is on record, and so that a future reader does not have to repeat
  the analysis.
- Any change to the physical models. The energy model stays as it is, and the
  limitation it imposes on the energy objective stays documented in
  [Part 3, Literature Comparison](#part-3-literature-comparison). That was the
  position when the plan was written. Phase 5 changed the energy model anyway,
  and the change and its measurement are in section 10. The wear calibration and
  the reward scale changed as well, and both are in section 11.

---

### 8. Risks

| Risk | Mitigation |
| :--- | :--- |
| The conditioned policy needs more steps than a single-weight policy to cover the same weight range. | Train on a sampled weight space with the same budget used for the sweep, and compare against the measured sweep rather than against a guess. |
| Domain randomization slows learning, because every episode is different. | Measure against the current policy on the same budget before keeping the change. |
| The receding-horizon expert is slow enough to dominate the collection cost. | The expert solves a three-variable problem per pass; if it turns out to be the bottleneck, cache its solutions by state. |
| Fine-tuning numbers depend on the episode budget. | The budget is a parameter of the experiment and is reported with the result. |

---

### 9. Execution order

| Phase | Effort | Unblocks |
| :--- | :--- | :--- |
| 1. Domain randomization | 1.5 h | The robustness claim, phase 2 |
| 2. Conditioned policy | 3 h | Retires the 21-run sweep |
| 3. Transfer experiment | 2 h | A deployment story |
| 4. Controller teacher | 4 h | A fair classical comparison |

Each phase lands as its own commit with tests, and each produces an artifact
that the documentation quotes.

---

### 10. What was built, and what it measured

| Phase | Built in | Artifact | Outcome |
| :--- | :--- | :--- | :--- |
| 1. Domain randomization | `MillingDT/config/scenarios.py`, `MillingDT/env/milling_env.py` | `results/run_PPO_seed42/metrics/monte_carlo.csv` | The finished surface stays inside the roughness limit in 96% of the 100 scenarios, the part completion rate is 94.0%, the flank wear limit is never exceeded. |
| 2. Conditioned policy | `MillingDT/env/milling_env.py`, `MillingDT/env/__init__.py`, `experiments/train_rl.py`, `experiments/evaluate.py` | `results/run_PPO_seed42/metrics/weight_sweep.csv`, `results/evaluation/weight_sweep_vs_ablation.csv` | One run of 1,000,000 steps against the 21 weight-specific policies of 150,000 steps each, which is 3,150,000 steps in total: 1.4% to 5.6% more removal rate for 2.9% to 6.1% less energy over the weights 0.0 to 0.6, and 9.0% and 21.6% less removal rate at 0.8 and 1.0. The earlier claim that the one policy beat all 21 by a wide margin is reversed under the current reward. |
| 3. Transfer experiment | `experiments/transfer_experiment.py` | `results/evaluation/transfer_results.csv` | Four arms. Fine tuning the source policy for 50,000 steps adapts it (168.5 to 101.9, with the roughness violations down to a quarter of them), and a policy trained on the target from scratch for the same budget lands in the same place (101.9). The production policy needs no adaptation at all and scores 190.1, the best of the four. |
| 4. Controller teacher | `MillingDT/baselines/receding_horizon.py` | `results/evaluation/baselines.csv` | The receding-horizon controller scores 253.5 against 10.0 for the constant-high policy, 5.1 for NSGA-II and 3.3 for the rule-based policy, and -1882.9 for constant safe. It finishes inside the roughness safety limit, at 1.25 um. |
| 5. Speed dependent cutting energy | `MillingDT/config/physics.py`, `MillingDT/config/physics_constants.yaml` | `results/evaluation/ablation_summary.csv` | The part energy now has an interior minimum in the spindle speed, and the ablation measures a real trade-off: across the non-dominated weights the removal rate falls 41% and the energy of the part falls 17%. |
| 6. Controller initialization | `MillingDT/agent/controller_init.py`, `experiments/controller_experiment.py` | `results/evaluation/controller_init_results.csv` | Three arms on the same 100 randomized scenarios at $w_2 = 0.4$. The initialization is actively harmful: the initialized policy cuts a part in 42.6 s against the controller's 17.1 s and the from-scratch policy's 11.4 s, and it exceeds the flank wear limit on 9% of the parts, which the other two arms never do. |
| 7. Reward scale, controller and wear calibration | `MillingDT/config/physics_constants.yaml`, `MillingDT/baselines/receding_horizon.py` | `results/evaluation/robustness_results.csv`, `results/evaluation/controller_init_results.csv` | The three defects of the comparison against the controller, fixed and re-measured: the reward counts the throughput, the controller optimizes the same expected episode reward, and the wear coefficient is anchored on a tool life. Section 11. |
| 8. Robustness experiment | `experiments/robustness_experiment.py` | `results/evaluation/robustness_results.csv` | Three regimes. Inside a perfect digital twin the controller scores 248.9 against the policy's 233.6, a gap inside the noise. With a wrong model of the material the controller falls to 63.5 and the policy holds at 226.9. With a noisy wear measurement the controller falls to 181.3 and the policy trained with the same noise keeps 213.6. Section 12. |
| 9. Batch experiment | `MillingDT/env/milling_batch_env.py`, `experiments/batch_experiment.py` | `results/evaluation/batch_results.csv` | A batch of ten parts cut with one tool, over 30 batches. The batch context is worth 24% fewer tool changes and 9.5% less energy at the same completion, and the batch-aware policy cuts the batch in 113.1 s against the per-part controller's 207.5 s. Section 13. |
| 10. Reward sensitivity | `experiments/reward_sensitivity.py` | `results/evaluation/reward_sensitivity.csv` | How much of the comparison against the controller the reward decides. The reward credits the removal rate of every pass, so the controller, which takes 5.85 passes against the policy's 4.00, collects 27% more objective credit for the same material, and about nine of the fifteen points between the two arms in the ideal case are that artifact. Part 1, section 5, and section 5.9 of Part 3. |
| 11. Regime map | `experiments/regime_map.py` | `results/evaluation/regime_map.csv` | Where the boundary between the two methods is, swept two ways: a noisy wear measurement and a miscalibrated material model. The crossover is at about 2% on both of them: below it the classical controller is the better operator, above it the learned policy is. The controller falls 31% from 248.9 to 171.5 as the noise grows, the policy 5% from 233.6 to 222.0, and training for the noise does not pay. The policy holds to a 20% model error and has a limit of its own at about 20% to 25%. Section 12. |

Four results contradict what the plan expected, and they are reported as they
came out.

**Fine tuning does adapt, and it is still not the best way to change
material.** The plan assumed that a pre-trained policy would adapt to a new
material faster than one trained from scratch, and an earlier measurement said
the opposite: fine tuning the source policy for 50,000 steps left it where it
was, with the roughness violations untouched. That
measurement was taken with the 8-dimensional observation, where the policy
could not see the material at all, so there was nothing for it to adapt with.
The measurement has been redone on the 11-dimensional observation, and the
adaptation holds: fine tuning reaches 101.9 with the roughness violations down
to a quarter of them, level with a policy trained on the target from scratch
for the same 50,000 steps. Both score below the production policy, which
needs no adaptation step at all and scores 190.1 on the target material.
The reason the plan's own recommendation still holds is phase 1: a
policy trained across a distribution of materials needs no adaptation step at
all.

The other surprise was the opposite one. The plan expected the energy
objective to be a trade-off against throughput and found that it was not: the
part energy was fixed by the volume removed, so both objectives pointed the
same way. That was a property of the energy model, not of the method, and
phase 5 changed the model to $k_c(v_c) V + P_{\text{idle}} T_{\text{part}}$
so that the trade-off exists to be found. It is now measured rather than
assumed.

The third is the controller initialization. The plan expected that starting
the policy from the controller would give a policy that beats the expert it
started from. Measured over three arms on the same 100 randomized scenarios,
it does not: under the current reward the initialization is actively harmful,
because it starts the policy at the conservative operating point and the wear
constraint makes that conservatism expensive. Section 11 has the measurement
and the reason.

The fourth is the strength of the classical controller. The plan assumed the
controller was a teacher, a source of better behaviour to clone. It turned out
to be a fair baseline for the same objective, which is a stronger claim: it
means the learned policy has to beat a controller that optimizes exactly what
the environment pays, with the model and the state in its hand. Section 12
measures where that comparison is won and where it is lost.

---

### 11. The three defects in the comparison, and the three fixes

Three defects were found in the comparison between the learned policy and the
classical controller. All three are fixed, and every artifact was regenerated
from scratch.

#### 11.1 The reward did not count the throughput

The objective term, the throughput against the energy, was worth about 1.2
points of reward per part against a completion bonus of 200. The whole
trade-off was a rounding error in the reward, so a policy that finished every
part at any cost scored well, and the operating point it settled on had little
to do with the objectives the reward names.

The scales now live in `MillingDT/config/physics_constants.yaml`, in the
`reward` section, and the objective term is worth about fifty points per part,
a quarter of the value of a part. The completion bonus is the value of a part
and the objective term is set against it, so the bonus still dominates the
decision to finish while the throughput decides how the part is cut. The scale
is a design decision and the configuration says so.

#### 11.2 The classical controller was not a baseline

It maximized the objective term per pass and nothing else. It ignored the
breakage probability, the step penalty, the completion bonus and the Pareto
margin, it held the wear to `VB_max` where the reward penalizes any pass
above `VB_safe`, and it committed to one feed for the whole part while the
reward penalizes only the finishing pass.

It is now a part-level maximizer of the expected episode reward, with a
separate roughing and finishing feed and with the breakage risk priced. On the
same scenarios it went from 184.3 to 248.9.

#### 11.3 The wear calibration was inert

$C_{\text{wear}} = 1.1\times10^{-4}$ gave a tool life of about 870 parts, so the wear constraint
did nothing inside a part and the tool life was irrelevant to a batch. None of
the three reference papers carries a Taylor wear model, so the coefficient was
never calibrated against one. It is now anchored on the tool life that Wang et
al. report (Journal of Manufacturing Systems 78, 2025): 40 parts cut with 5
identical tools, about 8 parts per tool. `C_wear = 1.0e-2` gives about 9 parts
per tool at the reference cutting condition.

The consequence of the three fixes is that every conclusion the previous
revision drew from the comparison against the controller has to be measured
again, and it has been. The wear constraint binds now, which is what makes the
controller initialization harmful (section 6) and what makes the tool life a
resource in the batch experiment (section 13).

---

### 12. The robustness experiment and the regime map

#### The question

The controller solves every pass with the exact model and the exact state, so
inside a perfect digital twin it owns the throughput and the energy: it
computes the optimum of the same objective the policy is learning, with
strictly more information. The interesting question is not whether the policy
can beat it there, it is where that information advantage disappears. A
controller that is given a wrong model of the material, or a noisy reading of
the wear, is the situation a plant is actually in.

#### The design

`experiments/robustness_experiment.py` measures the two arms on the same 100
randomized scenarios at $w_2 = 0.4$, in three regimes:

1. The nominal machine. The reference comparison, the one where the controller
   has every advantage.
2. Model error. The machine runs a harder material than the controller was
   built for: the wear rate is 30% higher and the roughness coefficients are
   higher. The controller solves with the coefficients it believes, and the
   policy is measured both when it is told the material and when it is not.
3. A noisy measurement of the wear. The controller trusts the reading it is
   given, and the policy is trained with the same 5% noise on its observation.
   That policy is the run `results/run_PPO_noisystate_seed42`.

The artifact is `results/evaluation/robustness_results.csv` and the tables are
in section 5.7 of
[Part 3, Literature Comparison](#part-3-literature-comparison). The reading is
short: inside a perfect digital twin the controller scores 248.9 against the
policy's 233.6 and the gap is inside the noise, and about nine of those
fifteen points are the artifact of the reward, which credits the removal rate
of every pass (Part 1, section 5). With a wrong model of the material the
controller falls to 63.5 and the policy holds at 226.9. With a noisy wear
measurement the controller falls to 181.3 and the policy trained with the same
noise keeps 213.6.

#### The regime map: how wrong the setup's model of the machine has to be

The robustness experiment compares two situations. The regime map measures the
boundary between them, so the answer is a number a plant can act on. It has
two sweeps, and together they are one statement.

The first sweep degrades the measurement. `experiments/regime_map.py` adds a
Gaussian error of sigma to the measured flank wear and the measured remaining
stock, sigma expressed as a fraction of their range. The controller solves the
next pass from that reading; the policy reads the same noise on its
observation.

Nothing is tilted towards the policy, and two points are deliberate. Both arms
are the general ones, neither is trained for the noise. The policy that was
trained with the noise exists and is reported as a third, clearly labelled
arm, but it is not the headline comparison, because a controller can also be
prepared for a noisy instrument by widening its safety margin, and that would
be a design choice rather than a training one. And the noise is tilted against
the policy: the controller is given a noisy reading of the flank wear, the
measurement it depends on, while the policy receives noise on the flank wear
and on the remaining stock.

| Measurement noise | Receding-horizon | PPO | PPO trained for the noise | PPO ahead by |
| ---: | ---: | ---: | ---: | ---: |
| 0.0% | 248.9 | 233.6 | 219.6 | -15.3 |
| 1.0% | 242.3 | 233.4 | 216.9 | -8.9 |
| 2.0% | 231.5 | 233.1 | 214.3 | +1.7 |
| 3.0% | 211.5 | 230.0 | 214.1 | +18.5 |
| 5.0% | 181.3 | 226.8 | 213.6 | +45.5 |
| 7.5% | 179.1 | 223.2 | 207.9 | +44.1 |
| 10.0% | 171.5 | 222.0 | 206.5 | +50.6 |

Three things to state:

1. The crossover is at about 2% of the range of the wear measurement. Below it
   the classical controller is the better operator, above it the learned policy
   is. That is the actionable form of the whole result.
2. The two degrade differently. The controller falls from 248.9 to 171.5, a 31%
   loss, because a noisy wear reading makes its plan violate the limits and
   break tools. The policy falls from 233.6 to 222.0, a 5% loss, because a
   perturbed observation perturbs its action slightly. A model-based
   controller has a cliff, a learned policy has a slope.
3. Training for the noise does not pay. The policy trained with the noise is
   below the general policy at every noise level, including the 5% it was
   trained for (213.6 against 226.8). The robustness of the general policy
   comes from being trained across a distribution of scenarios, and training
   against the noise only makes the policy more conservative.

The second sweep degrades the model. The machine runs the material of the
scenario, and the setup's belief about that material is off by a systematic
fraction, optimistically: it under-rates the wear and the roughness by that
fraction, which is the damaging direction, because a setup that thinks its
tool is better than it is will push it harder. The same belief goes to both
arms, the controller as the model it solves with and the policy as the
material in its context.

The policy trained for the noise is reported in the third column, and it is
not the headline: it was trained for a noisy instrument and not for a wrong
model, and it is not the general-purpose operator the comparison is about.

| Model error | Receding-horizon | PPO | PPO trained for the noise | PPO ahead by |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 248.9 | 233.6 | 219.6 | -15.3 |
| 5% | 201.5 | 232.8 | 219.6 | +31.2 |
| 10% | 136.0 | 237.8 | 219.6 | +101.8 |
| 20% | 35.7 | 201.5 | 219.6 | +165.8 |
| 30% | 32.5 | 116.7 | 219.6 | +84.3 |
| 40% | 39.3 | 117.0 | 219.6 | +77.7 |

Four things to state:

1. The controller's tolerance of a wrong model is about 2% to 3%, the same
   order as its tolerance of a noisy measurement. A 5% miscalibration costs it
   47 points (248.9 to 201.5) and a 10% error costs it 113 points (to 136.0).
   It collapses to 35.7 at 20%.
2. The policy is unaffected up to 20%: 201.5 at a 20% error, which is 13%
   below its own no-error score, and it is marginally better than at zero
   error at 5% and 10% (232.8 and 237.8 against 233.6, a difference inside the
   noise). It collapses at 30% (116.7), and the reason is that the belief has
   then left the distribution the policy was trained on, so its context
   feature is clipped at the edge of the band.
3. A conservative policy is insensitive to a wrong model. The policy trained
   for the noise is flat at 219.6 across the whole range, because it keeps the
   tool away from its limit whatever it believes about the material. It pays
   for that with the reward, and it is below the general policy everywhere.
4. So the two sweeps are one statement. The learned policy is the better
   operator whenever the setup's model of the machine is wrong by more than
   about 2%, whether that is a wear measurement noisier than 2% of its range or
   a material model miscalibrated by more than 2%. Below that the classical
   controller is better. The policy's own limit, around 20% to 25%, is where
   the belief leaves the distribution it was trained on, and that limit is a
   design parameter: widen the randomization band of the training and the
   tolerance widens with it.

The artifact is `results/evaluation/regime_map.csv` and the figure is
`results/plots/regime_map.png`, one panel per sweep.

#### The claim

The claim is not that reinforcement learning beats a model-based controller on
the objective inside a perfect digital twin. The two methods are comparable
there, and the comparison is biased against the learned policy, because the
reward pays the controller for the passes it takes. The claim is a boundary
with a number on it: the learned policy is the better operator once the
setup's model of the machine is wrong by more than about 2%, whether that is a
wear measurement noisier than 2% of its range or a material model
miscalibrated by more than 2%, and the classical controller is the better
operator below it. The policy has a limit of its own at about 20% to 25%,
where the belief leaves the distribution it was trained on, and that limit is
set by the randomization band of the training. The learned policy is also the
better throughput operator wherever the two compete. Of the three tests, the
noise sweep and the model error sweep are the fairer ones, because they
degrade one shared input instead of giving one method a different model of
the machine, while the model error regime of the robustness experiment asks
the controller to be wrong about a material and leaves the policy right. The
measurement of how much of the nominal gap the reward decides is section 5.9
of [Part 3, Literature Comparison](#part-3-literature-comparison).

---

### 13. The batch experiment and the batch environment

#### The question

The controller in this project solves one part at a time. It has no notion of
the tool, which is the one resource a batch of parts shares: the flank wear
carries from one part to the next, and a tool that reaches its limit has to be
replaced. A policy that sees the tool life and the batch that is left can
trade the speed of one part against the tool life the rest of the batch needs.

#### The environment

`MillingDT/env/milling_batch_env.py` holds `MillingDTBatchEnv`, registered as
`MillingDTBatchEnv-v1`. One episode is a batch of ten parts cut with one tool.
The observation is the eleven features of the part plus the tool life left and
the fraction of the batch still to cut. The tool is replaced when it breaks or
reaches its wear limit, and the replacement costs 500 reward units against the
200 a finished part is worth. The environment wraps the per-part environment and
adds nothing to the physics: it decides when a part is over, moves the tool
into the next part and keeps the count of the tool changes.

#### The design

`experiments/batch_experiment.py` measures three arms on the same 30 batches:
the batch-aware policy trained on the batch environment
(`results/run_PPO_batch_seed42`), the per-part policy applied part by part
inside the batch, and the per-part controller applied part by part inside the
batch. The difference between the first and the second is what the batch
context is worth.

The artifact is `results/evaluation/batch_results.csv` and the table is in
section 5.8 of
[Part 3, Literature Comparison](#part-3-literature-comparison). The reading is
short: the batch context is worth 24% fewer tool changes and 9.5% less energy
at the same completion, and the per-part controller needs 207.5 s of machine
time for the batch against the batch-aware policy's 113.1 s.

---

## Part 3: Literature Comparison


---

This document positions the energy-aware CNC milling digital twin in this
repository against the three papers held in `docs/papers/`.

Every number below is either quoted from one of those papers, with the section
it comes from, or measured by a script in this repository, with the artifact
path. Nothing is estimated, and no result is carried over from a run that is
no longer in the repository.

An earlier version of this document attributed these papers to "Gao et al.",
"Ferrisi et al." and "Zhou et al.", and carried a benchmark table whose
literature columns could not be traced to the papers, together with a
full column of results for a Soft Actor-Critic policy that no script in this
repository has ever produced. Those attributions, those columns and those
numbers are removed here.

---

### 1. Sources

| Key | Citation | Local copy |
| :--- | :--- | :--- |
| **Mo et al. (2025)** | X. Mo, M. Xia, S. Mei, X. Hu, "Deep reinforcement learning-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", *Journal of Manufacturing Processes* 156 (2025) 1006-1031. | `docs/papers/DRL-based optimization ...pdf` |
| **Wang et al. (2025a)** | P. Wang, Y. Cui, H. Tao, X. Xu, S. Yang, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", *Journal of Manufacturing Systems* 78 (2025) 124-152, doi 10.1016/j.jmsy.2024.11.013. | `docs/papers/Machining parameter optimization ...pdf` |
| **Wang et al. (2025b)** | P. Wang, Y. You, W. Liu, D. Ding, F. Xu, "Knowledge-guided deep reinforcement learning framework for feature machining step sequencing", *Int. J. Adv. Manuf. Technol.* 140 (2025) 5617-5636, doi 10.1007/s00170-025-16606-4. | `docs/papers/s00170-025-16606-4.pdf` |

---

### 2. What each work actually does

| Dimension | Mo et al. (2025) | Wang et al. (2025a) | Wang et al. (2025b) | This work |
| :--- | :--- | :--- | :--- | :--- |
| Process | Multi-pass turning of aero-engine casings | Batch milling of thin-walled aluminium parts | Feature machining step sequencing (process planning) | Multi-pass face milling |
| RL algorithm | PPO with a discrete tool-change head (DRL-TCDDO) | Multi-task TD3 (MATD3) | Knowledge-guided DRL with action masking | PPO |
| Action space | Tool change, feed, and other cutting parameters, continuous plus one discrete | Continuous (n, f, a_p) | Discrete feature step sequence | Continuous (n, f_r, a_p) |
| Objectives | Machining efficiency and quality, scored with HV, GD, IGD, SP | Surface roughness and machining efficiency, with tool deterioration estimated by multi-task learning | Sequencing cost and time | Material removal rate, energy, and hard constraints on roughness, wear and breakage |
| Tool condition model | Multi-task teacher-student CNN over acoustic emission signals (MTTS-TCM), plus a deep-learning roughness estimator | Data-driven multi-task time-series estimator of tool wear | Not applicable, the tool state is not a decision variable | Analytic Taylor-expanded wear model, no sensor pipeline |
| Validation | Two machining cases against 11 metaheuristics, plus a test on a new machine configuration | CNC dataset of thin-walled parts, batch experiments | Benchmark on feature sequencing problems | 100-scenario Monte Carlo sweep with 95% confidence intervals, and a 7-point energy weight ablation over 3 seeds |

Energy appears in the related-work discussion of Mo et al. (2025) and of
Wang et al. (2025a), but it is not one of their optimization targets. In Wang
et al. (2025b) the decision being made is a sequence of machining steps, so
continuous cutting parameters are outside its scope.

---

### 3. Published operating envelopes

Quoted from Mo et al. (2025), Table 5 and Section 3.4, and from Wang et al.
(2025a) for the tool life. The third column is
this repository, from `MillingDT/config/physics_constants.yaml`.

| Quantity | Published | This work |
| :--- | :--- | :--- |
| Cutting speed | Mo et al. (2025): 25 to 200 m/min | 314 to 1257 m/min at D = 50 mm, from 2000 to 8000 RPM |
| Feed per revolution | Mo et al. (2025): 0.05 to 0.3 mm/r | 0.05 to 0.35 mm/r |
| Depth of cut | Mo et al. (2025): 0.05 to 3 mm | 0.1 to 2.5 mm |
| Machining allowance | Mo et al. (2025): 3, 5, 6 or 10 mm | 5 mm nominal, 2 to 15 mm across the sweep |
| Surface roughness limit | Mo et al. (2025): Ra 3.2 um roughing, Ra 1.6 um finishing | Ra 1.6 um, 1.36 um with the 0.85 safety factor |
| Maximum spindle speed | Mo et al. (2025): 2000 RPM (TC-800), 4000 RPM for the new configuration | 8000 RPM |
| Tool wear limit | Mo et al. (2025): separate roughing and finishing thresholds, plus a breakage state | Flank wear VB 0.3 mm, 0.255 mm with the safety factor |
| Tool life | Wang et al. (2025a): 40 parts cut with 5 identical tools, about 8 parts per tool | About 9 parts per tool at the reference cutting condition, from the calibrated wear coefficient |

The last row is the anchor of the wear coefficient. None of the three papers
carries a Taylor wear model, so the coefficient in
`MillingDT/config/physics_constants.yaml` is not taken from any of them. It is
calibrated against the tool life that Wang et al. (2025a) report, so that the
wear constraint binds inside a part and the tool life is a resource a batch
can run out of.

---

### 4. What this work adds, and what it does not

**Adds**

1. Energy as an explicit objective. The reward carries a specific cutting
   energy term, `E = (P_idle + k_c(v_c) * MRR) * t_cut`, with the specific
   cutting energy growing with the cutting speed, and the energy weight `w2`
   is a first-class parameter of the environment rather than a by-product.
2. No sensor machine learning pipeline. Tool wear and surface roughness are
   propagated from the constants in `MillingDT/config/physics_constants.yaml`, so
   a run is reproducible from its seed and carries no training-data dependency.
   The wear coefficient is anchored on the tool life that Wang et al. (2025a)
   report, 40 parts cut with 5 identical tools, because none of the three papers
   carries a Taylor wear model that could calibrate it.
3. Statistical robustness reporting. Every metric in the Monte Carlo sweep
   comes with a t-distribution 95% confidence interval over 100 randomized
   scenarios, covering workpiece geometry, material and tool coefficients, and
   the initial flank wear of the tool.
4. An explicit energy ablation. Seven energy weights from 0.0 to 1.0, three
   seeds each, all evaluated on the same scenario set, produce the empirical
   throughput-energy frontier in `results/plots/ablation_pareto_frontier.png`.
5. A conditioned policy. The observation carries the energy weight and the
   material context, and both are drawn per episode, so one policy covers the
   whole weight range instead of one policy per weight. The condition lives in
   the one registration for the part, so there is no second environment to keep
   in step.
6. A receding-horizon controller as a classical baseline. The static NSGA-II
   baseline cannot react to the tool state, and three of the four static
   baselines finish outside the roughness safety limit, while the fourth takes
   768 s per part. The receding-horizon one is a part-level maximizer of the
   expected episode reward, it re-plans at every pass with a separate roughing
   and finishing feed, and it is the only classical policy that stays inside
   the limit.
7. A transfer experiment. A policy trained on one material is evaluated on a
   harder one, zero shot, then fine tuned, and compared against a policy
   trained on the new material from scratch for the same budget. The
   production policy, which has the material in its observation, is measured
   on the same target, and it needs no adaptation at all.
8. A controller initialization experiment. The receding-horizon controller is
   recorded on 2000 randomized episodes, a supervised actor with the shape of
   the PPO actor is fitted on those trajectories, and the fitted weights are
   copied into a fresh policy before the reinforcement learning starts. The
   resulting policy is measured against the controller and against a policy
   trained from scratch, on the same 100 randomized scenarios.
9. A robustness experiment against the classical controller. The two arms are
   measured in three regimes: the nominal machine, a machine that cuts a
   material the controller has no model for, and a machine whose wear
   measurement is noisy. It is the measurement that says where the learned
   policy is ahead and where it is not.
10. A batch environment. One episode is a batch of parts cut with one tool, so
   the tool life is the resource the parts share and the tool change is paid
   for. The batch-aware policy is measured against the same policy without the
   batch context and against the per-part controller.
11. A reward sensitivity study of the comparison against the controller. The
   reward credits the removal rate of every pass, so a plan that cuts the same
   part in more, shallower passes collects more credit. The two arms are
   measured over a range of objective scales and under a second form of the
   objective term, and the measurement shows that the ideal case comparison
   is biased against the learned policy by about nine of the fifteen points
   that separate the two arms.

12. A regime map of the comparison against the controller. Two sweeps, and
    together they are one statement. The noise on the wear measurement is
    swept from 0% to 10% of its range, and the error of the material model is
    swept from 0% to 40%, with the controller, the general policy and the
    noise-trained policy measured at every level. The crossover is measured
    at about 2% on both of them: below it the classical controller is the
    better operator, above it the learned policy is. The policy has a limit
    of its own at about 20% to 25% of model error, where the belief leaves
    the distribution it was trained on, and that limit is set by the
    randomization band of the training. The artifact is
    `results/evaluation/regime_map.csv`.

**Does not add**

1. No Soft Actor-Critic implementation. The repository trains and evaluates
   PPO only. An earlier version of this document reported SAC results with a
   16.4% energy reduction; no script here produced those numbers and the
   claim is withdrawn.
2. No tool replacement decision inside a part. The policy controls cutting
   parameters for a single tool life. The tool change is decided by the batch
   environment when the tool breaks or reaches its wear limit, and the policy
   sees the tool life and the batch that is left.
3. No real machine validation. The digital twin is calibrated against
   published models and against the tool life Wang et al. (2025a) report, not
   against measured cuts from a machining centre.
4. No learned surface roughness or tool condition estimator.

#### What the energy objective trades against

The energy of a finished part in this digital twin is

$$E_{\text{part}} = k_c(v_c) \cdot V + P_{\text{idle}} \cdot T_{\text{part}}$$

where $V$ is the volume that has to come off, $T_{\text{part}}$ is the cycle
time, and the specific cutting energy grows with the cutting speed:

$$k_c(v_c) = k_{c0} \left(\frac{v_c}{v_{\text{ref}}}\right)^{m}$$

This is the mechanism that makes the energy objective a real objective. The
material removal rate is inversely proportional to the cycle time,
$MRR = V / T_{\text{part}}$, so raising the spindle speed raises the removal
rate and cuts the idle energy, but it also raises the energy per unit volume
removed. The part energy therefore has an interior minimum in the spindle
speed, and the energy weight chooses where on that curve the policy sits:
a high energy weight holds the spindle back and gives up removal rate, a low
one runs it up and pays for it in energy.

Before this model was changed, the cutting term was fixed by the volume and
both terms moved the same way, so the energy objective was a duplicate of the
throughput objective and no weight on it could open a trade-off. The
ablation in section 5.2 is the measurement of the change: the removal rate and
the part energy now move in opposite directions across the weight grid.

Two things still bound the operating point, and neither of them is energy:

1. The surface roughness limit. A policy that pushes the feed rate up to raise
   the removal rate raises the roughness with it, and the roughness limit is
   what stops it.
2. The flank wear limit, which caps how long the tool survives the cycle.

The reward charges the energy of a pass, while the reported metric is the
energy of a part. A part cut in many shallow passes draws less power per pass
but spends more time on the idle machine, so the two quantities can move in
opposite directions. The ablation reports both.

The trade-off is measured, not assumed. Across the non-dominated weights of
the ablation the removal rate falls 41% (1556.6 to 918.5 mm3/min) and the
energy of the part falls 17% (69.1 to 57.2 kJ), and the spindle speed falls
with them, from 7859 RPM to 5382 RPM. The direction of the spindle speed is
the mechanism the model predicts: a higher energy weight holds the spindle
back, the specific cutting energy stops growing with the speed, and the part
gets cheaper at the cost of the material removed. Section 5.2 has the grid.


---

### 5. Measured results from this repository

Every table below is generated from the artifacts by
`python -m experiments.summarize_results`, which writes the same content to
`results/RESULTS.md`. The one exception is marked where it occurs: the
confidence intervals of the rate rows, which are given as fractions here.

#### 5.1 Production policy, Monte Carlo sweep over 100 scenarios

Artifact: `results/run_PPO_seed42/metrics/monte_carlo.csv`

The policy is `results/run_PPO_seed42`, the production policy, trained for
1,000,000 steps with the scenario and the energy weight drawn per episode.
It is queried here at $w_2 = 0.4$.

| Metric | Mean | 95% CI |
| :--- | ---: | :--- |
| Material removal rate (mm3/min) | 1608.49 | 1499.22 to 1717.76 |
| Energy per part (kJ) | 66.40 | 56.87 to 75.93 |
| Cycle time per part (s) | 11.43 | 10.06 to 12.81 |
| Max surface roughness (um) | 1.610 | 1.575 to 1.645 |
| Max flank wear (mm) | 0.1407 | 0.1274 to 0.1540 |
| Parts completed within spec | 94.0% | |
| Roughness violations, worst pass | 55.0% | [0.452, 0.648] |
| Roughness violations, finished surface | 4.0% | [0.002, 0.078] |
| Roughness above the safety limit, worst pass | 87.0% | [0.804, 0.936] |
| Roughness above the safety limit, finished surface | 4.0% | [0.002, 0.078] |
| Flank wear violations | 0.0% | [0.000, 0.000] |
| Tool breakage | 6.0% | [0.013, 0.107] |

The confidence intervals of the rate rows are given as fractions of the
scenarios, the metric rows in their own units.

The scenarios randomize the stock allowance, the tool diameter, the cutting
length, the wear and roughness coefficients, and the initial flank wear of the
tool. The policy is trained on scenarios drawn from the same tables, so it is
measured on the distribution it was trained on. The mean operating point over
the 100 scenarios is 7907 RPM, 0.318 mm/r, 2.07 mm of depth of cut and 4.0
passes per part.

The worst pass of an episode exceeds the roughness limit in 55% of the
scenarios and the safety limit in 87%, and those are the roughing passes. The
reward constrains the finishing pass, which is the pass that sets the surface
of the part, and the finished surface is inside the limit in 96% of the 100
scenarios.

Under a reward that counts the throughput the policy is a much more aggressive
operator than it was, and it pays for that with the breakage rate and with the
finished surfaces: 6% of the parts break a tool and 4% of the finished surfaces
are above the roughness limit.

#### 5.2 Energy weight ablation, 7 weights and 3 seeds

Artifact: `results/evaluation/ablation_summary.csv`

The 21 policies live in `results/run_ablation_w2_{w2}_seed{seed}`. Each one has
its weight pinned, and each is measured at the weight it was trained for, on
the same 50 scenarios.

| w2 | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Spindle (RPM) | Feed (mm/r) | On the frontier |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.0 | 1397.1 +/- 14.8 | 70.4 +/- 0.1 | 14.8 +/- 0.3 | 7998 | 0.299 | dominated |
| 0.2 | 1456.9 +/- 14.6 | 70.0 +/- 0.1 | 13.8 +/- 0.2 | 7992 | 0.299 | dominated |
| 0.4 | 1524.9 +/- 40.2 | 69.3 +/- 0.2 | 13.0 +/- 0.3 | 7896 | 0.296 | dominated |
| 0.5 | 1556.6 +/- 9.7 | 69.1 +/- 0.2 | 12.7 +/- 0.1 | 7859 | 0.296 | yes |
| 0.6 | 1519.8 +/- 36.4 | 68.3 +/- 0.4 | 13.1 +/- 0.4 | 7638 | 0.296 | yes |
| 0.8 | 1224.4 +/- 11.3 | 63.0 +/- 0.2 | 15.4 +/- 0.1 | 6573 | 0.292 | yes |
| 1.0 | 918.5 +/- 14.9 | 57.2 +/- 0.2 | 20.1 +/- 0.4 | 5382 | 0.292 | yes |

This is the measurement the energy model was changed for. Across the frontier,
from $w_2 = 0.5$ to $w_2 = 1.0$, the removal rate falls 41% and the energy of
the part falls 17%, and the spindle speed falls with them, from 7859 RPM to
5382 RPM. That is the mechanism the model predicts: the specific cutting
energy grows with the speed, so the energy weight holds the spindle back and
pays for it in removal rate. The three lowest weights are dominated: they spend
more energy without removing more material.

#### 5.3 One policy against 21 weight-specific policies

Artifacts: `results/run_PPO_seed42/metrics/weight_sweep.csv` (the production
policy, 1,000,000 steps, asked for every weight at runtime) and
`results/evaluation/weight_sweep_vs_ablation.csv` (the comparison against
the 21 per-weight policies).

| w2 | MRR, one policy | MRR, per-weight | MRR gap | Energy, one policy | Energy, per-weight | Energy gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1475.9 | 1397.1 | +5.6% | 68.4 | 70.4 | -2.9% |
| 0.2 | 1518.0 | 1456.9 | +4.2% | 65.8 | 70.0 | -6.1% |
| 0.4 | 1608.5 | 1524.9 | +5.5% | 66.4 | 69.3 | -4.2% |
| 0.5 | 1630.3 | 1556.6 | +4.7% | 66.0 | 69.1 | -4.4% |
| 0.6 | 1540.7 | 1519.8 | +1.4% | 65.5 | 68.3 | -4.0% |
| 0.8 | 1114.5 | 1224.4 | -9.0% | 57.9 | 63.0 | -8.0% |
| 1.0 | 720.4 | 918.5 | -21.6% | 51.7 | 57.2 | -9.7% |

The weight is a runtime query, not a retrain: it travels in the observation.
One training run of 1,000,000 steps covers the whole weight range, against the
21 weight-specific policies of 150,000 steps each, which is 3,150,000 steps in
total. Over the weights 0.0 to 0.6 the single policy removes 1.4% to 5.6% more
material for 2.9% to 6.1% less energy; at 0.8 and 1.0 it removes 9.0% and
21.6% less.

That last row is a reversal of the earlier claim, which was that the single
policy beat all 21 by a wide margin. The weight in the observation is a weaker
signal than a dedicated policy at the two heaviest weights, where the trade-off
is steepest. The honest reading: one conditional policy covers the range and
is ahead over most of it, but a policy trained for one weight is better at the
extreme weights.

Swept on its own, the same policy gives 1475.9 mm3/min at $w_2 = 0.0$ and
720.4 at $w_2 = 1.0$, with the energy of the part falling from 68.4 kJ to
51.7 kJ and the spindle from 8000 RPM to 4111 RPM. The weight moves the
operating point in the direction the model predicts, and it does so without a
retrain.

#### 5.4 Classical baselines

Artifact: `results/evaluation/baselines.csv`

The same 50 episodes of the nominal condition, at $w_2 = 0.4$, for the five
classical policies.

| Policy | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Mean reward | Final Ra (um) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Receding horizon | 725.5 | 19.0 | 6.7 | 253.5 | 1.25 |
| Constant high | 1099.6 | 17.8 | 4.3 | 10.0 | 1.56 |
| NSGA-II static | 732.9 | 18.8 | 6.4 | 5.1 | 1.56 |
| Rule based | 719.8 | 18.8 | 6.6 | 3.3 | 1.56 |
| Constant safe | 1.6 | 385.7 | 768.0 | -1882.9 | 0.26 |

Three of the four static policies end outside the roughness safety limit at
1.56 um, which the reward penalizes with -50. The constant-safe policy stays
inside the limit at 0.26 um, and pays for it with 768 s and 386 kJ per part.
The receding-horizon controller is the strongest baseline in the repository, and
it is the only classical policy that stays inside the limit.

The learned policy is not in this table. It is measured against the controller
in section 5.7, on the scenarios the policy is trained and measured on, and
not on the nominal condition alone.

#### 5.5 Transfer to a harder material

Artifact: `results/evaluation/transfer_results.csv`

The target material is harder: $k_1 \times 1.25$, $k_2 \times 1.30$ and
$C_{\text{wear}} \times 2.0$, on the same machine. The source policy was
trained for 300,000 steps on the nominal material, and the adaptation budget
on the target is 50,000 steps. The measurement is over 30 scenarios.

| Arm | Mean episode reward | Roughness violations, worst pass | Roughness violations, finished surface | Parts completed |
| :--- | ---: | ---: | ---: | ---: |
| Source policy, zero shot | 168.5 | 100.0% | 6.7% | 86.7% |
| Source policy, fine tuned, 50,000 steps | 101.9 | 26.7% | 0.0% | 86.7% |
| Trained on the target from scratch, 50,000 steps | 101.9 | 26.7% | 0.0% | 86.7% |
| Production policy, zero shot | 190.1 | 100.0% | 10.0% | 90.0% |

The nominal-trained source policy violates the roughness limit on the worst
pass in every scenario. Fine tuning it for 50,000 steps, or training on the
target from scratch for the same budget, both reduce the violations to a
quarter of them: the adaptation budget is enough either way.

The production policy needs no adaptation at all. It was trained across a
distribution of materials with the material context in its observation, so the
harder material is inside its training distribution, and it scores the highest
reward of the four arms, 190.1, and completes the most parts. Its 100%
worst-pass violations sit on the roughing passes, which the reward does not
constrain.

The two adapted arms score below the source policy, 101.9 against 168.5, and
below the production policy. The reason is the harder material: it wears the
tool twice as fast, and the policy that slows down for that trades away more
throughput than it gains in specification.

#### 5.6 Starting the policy from the controller

Artifact: `results/evaluation/controller_init_results.csv`

Three arms, all on the same 100 randomized scenarios at the same energy weight
$w_2 = 0.4$: the receding-horizon controller itself, the production policy
trained from scratch with PPO, and a policy whose actor was initialized from
the controller and then trained with PPO for 1,000,000 steps.

| Arm | Mean episode reward | Reward std | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Ra violation, finished surface | Flank wear violations | Tool breakage | Parts completed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Receding-horizon controller | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 2% | 1% | 3% | 96% |
| PPO from scratch | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 4% | 0% | 6% | 94% |
| PPO initialized from the controller | 119.8 | 178.7 | 1274.9 | 81.5 | 42.6 | 4% | 9% | 4% | 87% |

The initialization is actively harmful, and the reason is instructive. The
cloned actor starts the policy at the controller's operating point, which is
the conservative one: a low feed and a low spindle speed so that every pass of
the part stays inside the roughness safety limit. With the wear constraint now
binding, that conservatism is expensive. The policy then learns from a bad
starting point, slows down further to stay away from the wear limit, and ends
up cutting a part in 42.6 s against the controller's 17.1 s and the
from-scratch policy's 11.4 s. It also exceeds the flank wear limit on 9% of
the parts, which the other two arms never do, and it completes the fewest
parts of the three.

An earlier revision of this document, measured under the old reward, found the
initialization harmless: it made the policy more consistent and did not change
the reward. That was the reward not counting the cost of the slow operating
point. With the throughput counted, the initialization is what the measurement
says it is. The method to keep from the controller initialization is the
record and clone machinery, not the initialization itself.

#### 5.7 Robustness against the classical controller

Artifact: `results/evaluation/robustness_results.csv`

The two arms on the same 100 randomized scenarios at the same energy weight
$w_2 = 0.4$, in three regimes.

| Regime | Arm | Mean reward | Reward std | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Tools broken | Parts completed |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The controller has every advantage | Receding-horizon controller | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 3.0% | 96.0% |
| The controller has every advantage | PPO | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 6.0% | 94.0% |
| The controller's material model is wrong | Receding-horizon controller | 63.5 | 117.1 | 1306.9 | 71.0 | 23.3 | 2.0% | 95.0% |
| The controller's material model is wrong | PPO | 226.9 | 88.3 | 1604.2 | 66.3 | 11.5 | 6.0% | 94.0% |
| The wear measurement is noisy | Receding-horizon controller | 181.3 | 185.9 | 1300.4 | 74.2 | 31.1 | 4.0% | 91.0% |
| The wear measurement is noisy | PPO trained with the noise | 213.6 | 87.3 | 1554.2 | 66.9 | 12.0 | 6.0% | 94.0% |

The artifact holds a third row for the model error regime, the policy that is
told the material: 215.1 mean reward, 94.2 std, 1548.7 MRR, 66.8 kJ, 12.2 s,
6.0% broken, 94.0% completed. It says that the policy does not need the
material to be told: the policy that believes the nominal material, 226.9, is
slightly better than the one that is told the truth, 215.1.

Four readings:

1. Inside a perfect digital twin the two methods are comparable, and the
   measured gap is partly the reward's artifact. The controller plans the part
   with the exact model, the exact state and the material in its hand, and it
   scores 248.9 against the policy's 233.6. The gap is inside the noise: the
   standard error of the mean is about 8 for both arms, so a difference of 15
   has a t of about 1.4. About nine of those fifteen points come from the
   reward crediting the removal rate of every pass: the controller takes 5.85
   passes against the policy's 4.00, so it collects 27% more objective credit
   for the same material. Section 5.9 is the measurement of that, and section
   5.7.1 is the measurement of where the sign of the comparison changes.
2. The learned policy is the better throughput operator. It removes 22% more
   material per minute (1608 against 1318), finishes a part in a third less
   time (11.4 against 17.1 s) and spends 3.7% less energy. It pays for that
   with twice the tool breakage, 6% against 3%, and 4% of its finished
   surfaces above the roughness limit against 2%.
3. The advantage appears where a plant actually is. With a material the
   controller has no model for, its reward collapses by three quarters, from
   248.9 to 63.5, because it does not slow down and its finished surfaces go
   out of specification. The learned policy holds at 226.9.
4. The same holds for a noisy measurement. With 5% noise on the wear and the
   remaining stock the controller's reward falls to 181.3 and its spread more
   than doubles, to 185.9 against the policy's 87.3. The policy trained with
   the same noise keeps 213.6 and cuts the part in 12.0 s against the
   controller's 31.1 s.

#### 5.7.1 How wrong the setup's model of the machine has to be

Artifact: `results/evaluation/regime_map.csv`
Figure: `results/plots/regime_map.png`, one panel per sweep

The robustness experiment compares two situations. The regime map measures the
boundary between them. It has two sweeps, and together they are one
statement.

The first sweep degrades the measurement. The flank wear and the remaining
stock are measured with an instrument, and the reading carries a Gaussian
error of sigma, a fraction of their range. The controller solves the next pass
from that reading; the policy reads the same noise on its observation.

Nothing is tilted towards the policy, and two points are deliberate. Both arms
are the general ones, neither is trained for the noise. The policy trained
with the noise is reported as a third, clearly labelled arm, and it is not the
headline comparison, because a controller can also be prepared for a noisy
instrument by widening its safety margin, and that would be a design choice
rather than a training one. And the noise is tilted against the policy: the
controller is given a noisy reading of the flank wear, the measurement it
depends on, while the policy receives noise on the flank wear and on the
remaining stock.

| Measurement noise | Receding-horizon | PPO | PPO trained for the noise | PPO ahead by |
| ---: | ---: | ---: | ---: | ---: |
| 0.0% | 248.9 | 233.6 | 219.6 | -15.3 |
| 1.0% | 242.3 | 233.4 | 216.9 | -8.9 |
| 2.0% | 231.5 | 233.1 | 214.3 | +1.7 |
| 3.0% | 211.5 | 230.0 | 214.1 | +18.5 |
| 5.0% | 181.3 | 226.8 | 213.6 | +45.5 |
| 7.5% | 179.1 | 223.2 | 207.9 | +44.1 |
| 10.0% | 171.5 | 222.0 | 206.5 | +50.6 |

Three things to state:

1. The crossover is at about 2% of the range of the wear measurement. Below it
   the classical controller is the better operator, above it the learned policy
   is. That is the actionable form of the whole result.
2. The two degrade differently. The controller falls from 248.9 to 171.5, a 31%
   loss, because a noisy wear reading makes its plan violate the limits and
   break tools. The policy falls from 233.6 to 222.0, a 5% loss, because a
   perturbed observation perturbs its action slightly. A model-based
   controller has a cliff, a learned policy has a slope.
3. Training for the noise does not pay. The policy trained with the noise is
   below the general policy at every noise level, including the 5% it was
   trained for (213.6 against 226.8). The robustness of the general policy
   comes from being trained across a distribution of scenarios, and training
   against the noise only makes the policy more conservative.

The second sweep degrades the model. The machine runs the material of the
scenario, and the setup's belief about that material is off by a systematic
fraction, optimistically: it under-rates the wear and the roughness by that
fraction, which is the damaging direction, because a setup that thinks its
tool is better than it is will push it harder. The same belief goes to both
arms, the controller as the model it solves with and the policy as the
material in its context.

The policy trained for the noise is reported in the third column, and it is
not the headline: it was trained for a noisy instrument and not for a wrong
model, and it is not the general-purpose operator the comparison is about.

| Model error | Receding-horizon | PPO | PPO trained for the noise | PPO ahead by |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 248.9 | 233.6 | 219.6 | -15.3 |
| 5% | 201.5 | 232.8 | 219.6 | +31.2 |
| 10% | 136.0 | 237.8 | 219.6 | +101.8 |
| 20% | 35.7 | 201.5 | 219.6 | +165.8 |
| 30% | 32.5 | 116.7 | 219.6 | +84.3 |
| 40% | 39.3 | 117.0 | 219.6 | +77.7 |

Four things to state:

1. The controller's tolerance of a wrong model is about 2% to 3%, the same
   order as its tolerance of a noisy measurement. A 5% miscalibration costs it
   47 points (248.9 to 201.5) and a 10% error costs it 113 points (to 136.0).
   It collapses to 35.7 at 20%.
2. The policy is unaffected up to 20%: 201.5 at a 20% error, which is 13%
   below its own no-error score, and it is marginally better than at zero
   error at 5% and 10% (232.8 and 237.8 against 233.6, a difference inside the
   noise). It collapses at 30% (116.7), and the reason is that the belief has
   then left the distribution the policy was trained on, so its context
   feature is clipped at the edge of the band.
3. A conservative policy is insensitive to a wrong model. The policy trained
   for the noise is flat at 219.6 across the whole range, because it keeps the
   tool away from its limit whatever it believes about the material. It pays
   for that with the reward, and it is below the general policy everywhere.
4. So the two sweeps are one statement. The learned policy is the better
   operator whenever the setup's model of the machine is wrong by more than
   about 2%, whether that is a wear measurement noisier than 2% of its range or
   a material model miscalibrated by more than 2%. Below that the classical
   controller is better. The policy's own limit, around 20% to 25%, is where
   the belief leaves the distribution it was trained on, and that limit is a
   design parameter: widen the randomization band of the training and the
   tolerance widens with it.

The claim is not that reinforcement learning beats a model-based controller on
the objective inside a perfect digital twin, because the controller computes
the optimum of the same objective with strictly more information, and because
the reward pays the controller for the passes it takes. In the ideal case the
two methods are comparable, with the policy ahead on the physical metrics and
behind on the tool breakage. The claim is a boundary with a number on it: the
learned policy is the better operator once the setup's model of the machine is
wrong by more than about 2%, whether that is a wear measurement noisier than
2% of its range or a material model miscalibrated by more than 2%, and the
classical controller is the better operator below it. The policy has a limit
of its own at about 20% to 25%, where the belief leaves the distribution it
was trained on, and that limit is set by the randomization band of the
training. The learned policy is also the better throughput operator wherever
the two compete. Of the three tests, the noise sweep and the model error
sweep are the fairer ones, because they degrade one shared input instead of
giving one method a different model of the machine, while the model error
regime of section 5.7 asks the controller to be wrong about a material and
leaves the policy right.

#### 5.8 The batch against the part

Artifact: `results/evaluation/batch_results.csv`

A batch is ten parts cut with one tool. The tool is replaced when it breaks or
reaches its wear limit, and the replacement costs 500 reward units against the
200 a finished part is worth. Three arms over 30 batches at $w_2 = 0.4$.

| Arm | Batch reward | Std | Mean reward per part | Tool changes per batch | Parts cut | Batch time (s) | Batch energy (kJ) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO, batch aware | 1817.0 | 477.8 | 166.8 | 0.63 | 10.0 | 113.1 | 610.4 |
| PPO, per part | 1709.9 | 366.2 | 169.4 | 0.83 | 10.0 | 114.7 | 674.3 |
| Receding-horizon controller, per part | 1859.9 | 328.8 | 173.7 | 0.83 | 10.0 | 207.5 | 692.0 |

The batch context is worth 24% fewer tool changes and 9.5% less energy at the
same completion, and the batch-aware policy also collects the higher batch
reward, 1817.0 against 1709.9, at a lower reward per part. The per-part
controller gets the highest batch reward, but it needs 207.5 s of machine time
for the batch against the batch-aware policy's 113.1 s.

#### 5.9 The reward sensitivity of the comparison against the controller

Artifact: `results/evaluation/reward_sensitivity.csv`

The reward credits the removal rate of every pass, which is the standard way
to express a rate in a reward that is paid per pass. The total credit then
grows with the number of passes a plan takes, so a plan that cuts the same
part in more, shallower passes is paid more for it. The two arms are rolled
out on the same 100 randomized scenarios at $w_2 = 0.4$, and the reward terms
are accumulated in a form that lets the accounting be varied afterwards.

| | The controller | The policy |
| :--- | ---: | ---: |
| Passes per part | 5.85 | 4.00 |
| Objective credit collected | +0.413 | +0.326 |
| Mean reward | 248.9 | 233.6 |

The two plans remove the same stock allowance. The controller takes more
passes, so it collects 27% more objective credit for the same material, and at
the objective scale of the configuration that is about nine of the fifteen
reward points that separate the two arms in the ideal case. The ideal case
comparison is therefore biased against the learned policy, and the defensible
reading of it is that the two methods are comparable, with the policy ahead on
the physical metrics and behind on the tool breakage. The regime map of
`results/evaluation/regime_map.csv` puts the ideal case at the left end of a
boundary: the crossover is at about 2%, whether the wear measurement is
noisier than 2% of its range or the material model is miscalibrated by more
than 2%, and below it the classical controller is the better operator. The
policy has a limit of its own at about 20% to 25%, where the belief leaves
the distribution it was trained on.

Three things were checked:

1. The objective scale is not the lever. The controller wins at every scale
   from 5 to 4000, and the gap only grows with it, from 7.1 points at 5 to
   352 points at 4000. The sign never flips.
2. The part level alternative was implemented and rejected. Crediting the
   material of the part against the energy of the part removes the pass count
   from the accounting, and under it the gap at the current scale is 5.0
   points instead of 15.3, and the policy is ahead at the two largest scales.
   It is not the right fix, because the material a part gives up is fixed by
   its stock: the volume of a part is `pi * D * kappa * L_cut / 1000`,
   independent of every cutting parameter. The throughput weight would then be
   inert, the reward would be an energy objective with a throughput weight that
   does nothing, and the weight sweep would stop measuring what it claims to
   measure.
3. What a correct formulation would need. The removal rate of a part is the
   ratio of a fixed volume to the part's time, and a rate cannot be
   accumulated over a part without becoming a volume. A correct reward has to
   price the cycle time directly, or compare the part's average rate to a
   reference at the end of the part. Both change the meaning of the weights,
   and the second makes the objective sparse and the credit assignment poor.
   Neither was adopted, and the reward as shipped carries the artifact
   instead, documented rather than hidden.

---

### 6. Reproducing the numbers above

```bash
# The whole pipeline, in order
python -m experiments.run_pipeline

# The production policy: the scenario and the energy weight are drawn per episode
python -m experiments.train_rl --algo PPO --steps 1000000 --seed 42 --randomize --randomize-weight

# Monte Carlo sweep with confidence intervals
python -m experiments.evaluate

# The same policy asked for every point of the trade-off
python -m experiments.evaluate --sweep-weight

# The comparison against the per-weight policies
python -m experiments.evaluate --compare-only

# Energy weight ablation, 7 weights x 3 seeds
python -m experiments.train_rl_ablation --steps 150000 --scenarios 50 --slots 7

# Classical baselines
python -m experiments.run_baselines

# Transfer to a harder material
python -m experiments.transfer_experiment 300000 50000

# Start the policy from the controller, against the controller and from scratch
python -m experiments.controller_experiment 1000000 2000

# Robustness against the classical controller, in three regimes
python -m experiments.robustness_experiment

# The regime map: how wrong the setup's model of the machine has to be
python -m experiments.regime_map

# The batch-aware policy
python -m experiments.train_rl --env-id MillingDTBatchEnv-v1 --steps 1000000 --seed 42 --randomize --run-name PPO_batch_seed42

# The policy that faces a noisy wear measurement
python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight --obs-noise 0.05 --run-name PPO_noisystate_seed42

# The batch against the part
python -m experiments.batch_experiment

# The reward sensitivity of the comparison against the controller
python -m experiments.reward_sensitivity

# Rebuild the result tables
python -m experiments.summarize_results
```

---

### 7. Future work

1. Implement Soft Actor-Critic and re-run the energy weight ablation under it,
   to test whether the off-policy entropy regularizer finds a different
   operating point than PPO.
2. Deploy on a 3-axis machining centre with a power meter, and compare the
   measured energy per part against the digital twin prediction.
3. Extend the wear model from a single flank wear value to the separate
   roughing and finishing thresholds used by Mo et al. (2025).
4. Publish the reward sensitivity alongside the comparison against a
   model-based controller. The comparison is sensitive to how the reward pays
   for a rate: the reward as shipped credits the removal rate of every pass,
   and therefore pays for the number of passes a plan takes, which biases the
   ideal case comparison against the learned policy by about nine of the
   fifteen points that separate the two arms. The sensitivity has to be
   published alongside the result. A reward that prices the cycle time
   directly, or that compares the part's average rate to a reference at the
   end of the part, would remove the artifact, but both change the meaning of
   the weights and the second makes the objective sparse and the credit
   assignment poor.
5. Measure the wear measurement noise of a real machine, and cost the
   controller's alternative against the same boundary. The regime map answers
   the question for a synthetic instrument error, and the crossover sits at
   about 2% of the range of the wear measurement, or at a material model
   miscalibrated by more than 2%. A real machine has to be measured the same
   way, and the alternative to a learned policy, which is a wider safety
   margin on the controller, has to be priced against the same boundary
   rather than assumed. The regime map also has to be extended to the other
   imperfect measurements, the roughness reading in particular, since the
   reward constrains the finishing pass on the roughness.

   The tolerance of the policy is a design parameter rather than a property
   of the method, and the randomization band of the training is the practical
   lever a plant has over it. The policy's own limit is at about 20% to 25%
   of model error, where the belief leaves the distribution it was trained
   on; widen the band and the tolerance widens with it. How wide the band
   should be for a given machine is part of the same question.

---

## Part 4: Interactive Simulator Plan


---

**Project:** Energy-Aware Deep Reinforcement Learning for Dynamic CNC Milling Optimization  
**Course:** ME6324 - AI in Manufacturing (IIT Madras)  
**Deliverable:** Autonomous Digital Twin Simulator (`app.py`, `MillingDT/sim/machining_sim.py`)  
**Design Standard:** Anti-Slop Industrial Instrument Guidelines (`.agents/skills/antislop/`, `antislop-ui`, `frontend-ui-engineering`)  

---

### 1. Problem Statement & Anti-Slop Directive

Generic AI demos present superficial dashboards with neon purple gradients, floating cards, vague "AI score" bars, and no physics fidelity. 
This Digital Twin simulator is designed as an **authentic CNC Machine Tool Telemetry & Material Removal Workstation**:
- **Zero AI Buzzwords**: No "unlocking next-gen synergy" or "revolutionary optimization" copy. Every figure corresponds to standard ISO/CIRP manufacturing units: $\text{mm}$, $\text{RPM}$, $\text{mm/rev}$, $\mu\text{m}$, $\text{mm}^3/\text{min}$, $\text{kJ}$, and $\text{kW}$.
- **Direct Physics Linkage**: Every pass directly executes empirical Taylor-extended flank wear models, specific cutting power calculations, and surface roughness regressions parameterized from actual experimental literature.
- **Physical Material Removal Simulation (Layer by Layer)**: A visual representation of the cylindrical/billet workpiece cross-section, showing initial stock, tool pass cuts, remaining allowance $\kappa$, depth per pass $a_p$, and the resulting micro-surface topography.

---

### 2. Innovative Technical Features

#### A. 2D Cross-Sectional Material Removal & Tool Engagement Simulation
- **Stock Depletion Geometry**:
  - Initial stock profile: Cylindrical billet of diameter $D$ with total machining allowance $\kappa_{\text{total}}$.
  - Pass-by-pass progressive cutout: Each cut removes a rectangular layer of depth $a_p$ and cut length $L_{\text{cut}}$.
  - The remaining stock boundary updates in real-time.
  - Interactive chip-load / engagement diagram: Visualizes the tool entry, radial engagement, and axial depth of cut.

#### B. Micro-Topography & Surface Roughness Profile ($Ra$) Generator
- Generates the realistic feed-mark cusps profile based on tool nose radius $r_\epsilon$, feed per tooth $f_z$, and flank wear condition:
  $$y_{\text{cusp}}(x) \approx \frac{x^2}{8 r_\epsilon} + \text{stochastic chatter/wear perturbation}$$
- Visually shows the peak-to-valley profile alongside the ISO 1302 centerline average $Ra$, contrasting it with the $1.60\ \mu\text{m}$ ceiling.

#### C. Live ISO 6983 G-Code Block Generator
- Emits real, syntax-highlighted CNC G-code for Fanuc / Siemens controllers for the exact parameter sequence calculated by the DRL agent:
  ```gcode
  %
  O1001 (PPO AUTONOMOUS MILLING ADAPTIVE RUN)
  G21 G90 G94 (METRIC, ABSOLUTE, FEED PER MIN)
  G17 G40 G49 G80
  T01 M06 (D50 CORNER MILL)
  S7500 M03 (PPO COMPUTED SPINDLE SPEED)
  G00 X0. Y0. Z5.0
  (--- PASS 1: ROUGHING PASS ---)
  G00 Z-2.500 (PASS 1 DEPTH ap = 2.50 mm)
  G01 X100.0 F2400.0 (FEED RATE fr = 0.320 mm/rev)
  ...
  ```

#### D. Multi-Controller Head-to-Head Telemetry
- Simulates the exact same workpiece simultaneously under:
  1. **Dynamic PPO (Our Autonomous Policy)**
  2. **Static NSGA-II (Multi-Objective Pareto Knee)**
  3. **Rule-Based Heuristic (Industrial Feed/Speed Wear Derating)**
  4. **Constant High (Max Throughput Baseline)**
- Compares machining cycle time, total energy, final $Ra$, and tool degradation.

#### E. Industrial Design Palette & Clean Human Interface (Anti-Slop)
- Strict dark-slate/industrial palette inspired by Siemens Sinumerik & Heidenhain CNC control panels:
  - Base: Deep Slate (`#0f172a`), Dark Charcoal (`#1e293b`)
  - Accent: Precision Cyan (`#06b6d4`), Safety Amber (`#f59e0b`), Alarm Crimson (`#ef4444`), Compliance Emerald (`#10b981`)
- No gratuitous animations or purple glows. Real-time updates with clear error bounds and 95% confidence bounds.

---

### 3. Architecture & Code Structure

```
G:/My Drive/IITM/AI in Mfg/

 MillingDT/
     sim/
         __init__.py
         machining_sim.py      # Trajectory simulation, surface profile and G-code synthesis

 app.py                         # Streamlit interactive application
```

This plan was consolidated into this file, and the document it names no longer
exists separately. It is kept as the design record.

#### Module Specifications:
1. `MillingDT/sim/machining_sim.py`:
   - `load_ppo_policy(model_dir)`: Loads the production policy with its `VecNormalize` statistics.
   - `run_single_simulation(policy_type, kappa_total, VB_0, D, L_cut, w2, ppo_model_tuple)`: Executes multi-pass machining and records state vectors $(VB, \kappa, n, f_r, a_p, MRR, Ra, E, P_c)$. The energy weight is a query of the policy, so the same production policy serves every weight.
   - `generate_surface_micro_profile(final_ra, feed_rate, tool_radius)`: Generates feed-mark surface profile coordinates.
   - `generate_cnc_gcode(trajectory_df, part_id, program_num, tool_diameter)`: Converts the parameter trajectory into executable CNC G-code, with the diameter the trajectory was cut with written into the tool change line.

2. `app.py`:
   - Interactive input controls (workpiece diameter, cut length, total allowance, initial flank wear, green slider $w_2$).
   - Layout:
     - **Top Metrics Row**: 4 key indicators (Throughput MRR, Specific Energy, Surface Roughness $Ra$, Tool Wear $VB$).
     - **Tab 1: Physical Material Removal & Tool Path**: 2D workpiece stock depletion visualizer + dynamic cut layers.
     - **Tab 2: Surface Micro-Topography & Tool Wear**: Profile diagram + Flank wear progression vs. limit lines.
     - **Tab 3: Head-to-Head Controller Comparison**: Side-by-side performance cards and radar chart.
     - **Tab 4: Fanuc / ISO G-Code Program**: Syntax-highlighted, downloadable `.nc` file.

---

## Part 5: Physics References


---

This document consolidates the key differences between our Digital Twin implementation and the original reference papers, as well as the exact physics equations utilized in the environment.

### 1. Source of Physics Equations

The core physics equations driving our Digital Twin environment (`MillingDTEnv-v1`) are sourced almost entirely from the **"DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"** paper, with specific concepts adapted from **"Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning"** and standard manufacturing literature.

#### From "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"
*(Mo, Xia, Mei, Hu, Journal of Manufacturing Processes 156 (2025) 1006-1031)*
* **Cutting Speed ($v_c$):** $v_c = (\pi \times D \times n) / 1000$ *(Eq. 5)*
* **Feed per Minute ($f$):** $f = n \times f_r$ *(Eq. 6)*
* **Cutting Time ($t_{cut}$):** $t_{cut} = L_{cut} / f$ *(Eq. 7)*
* **Material Removal Rate (MRR):** $MRR = v_c \times a_p \times f_r$ *(Eq. 27)*
* **Tool Wear Degradation ($\Delta VB$):** The Taylor-extended wear model ($\Delta VB = C \cdot v_c^\alpha \cdot f_r^\beta \cdot a_p^\gamma \cdot t_{cut}$) is a standard manufacturing wear law, not taken from this paper. The paper's Eq. 17 is the wear *constraint* ($VBi \le VBmax$), and its virtual environment obtains the wear value from a trained data-driven model (Section 5.2) rather than from an analytic wear law.

#### From "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" / General Literature
*(Wang, Cui, Tao, Xu, Yang, Journal of Manufacturing Systems 78 (2025) 124-152)*
* **Surface Roughness ($Ra$):** The empirical regression formula we use ($Ra = k_1 f_r^2 + k_2 VB + k_3 / v_c$) is a standard traditional model that "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" uses as a comparative baseline (Eq. 56 in their paper). "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" utilizes a very similar mathematical structure for their baseline objectives. 

#### Standard Manufacturing Engineering (Our Novelty)
* **Energy Consumption ($E$):** $E = (P_{idle} + k_c(v_c) \times MRR) \times t_{cut}$, with the specific cutting energy growing with the cutting speed, $k_c(v_c) = k_{c0} (v_c / v_{ref})^m$. This is not explicitly in "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation". We added the specific cutting energy formulation to introduce the "Green Manufacturing" objective into the reward function, and the speed dependence to make it an objective in its own right: without it the energy of a part is fixed by the volume, and the energy term is a duplicate of the throughput term.

The energy model with the speed dependent cutting energy is this project's own
modelling choice. It is taken from neither of the two reference papers: neither
of them models the energy of a cut, so the equations, the calibrated
coefficient $k_{c0} = 2.5$ W$\cdot$min/mm$^3$ and the speed exponent $m = 0.6$
are ours, and they are stated here so that a reader can judge them against
the measurements in
[Part 3, Literature Comparison](#part-3-literature-comparison) rather than
against a citation.

---

### 2. Our Implementation vs. "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"

We are remarkably close on the **Reinforcement Learning and Physics** side, but we have abstracted away the heavy **Signal Processing and Deep Learning** components.

#### What we implemented identically (or very closely):
1. **The Core Digital Twin Physics:** Exact equations for calculating cutting speed, material removal rate (MRR), and the Taylor-extended tool wear degradation model.
2. **Stochastic Tool Breakage:** Probability-based catastrophic tool breakage logic (where pushing a worn tool too hard causes a sudden failure), forcing the agent to learn a safe strategy.
3. **Multi-Objective Reward:** Giving immediate penalties for violating limits ($VB_{max}$, $Ra_{max}$) and using a Hypervolume (HV) reward at the end of the workpiece to balance time and quality.
4. **Multi-Pass Logic:** The agent manages the depth of cut ($a_p$) to sequentially eat away at a total machining allowance (`kappa`), differentiating between roughing passes and the final finishing pass.

#### Where we differ (What we abstracted away):
1. **No Acoustic Emission (AE) Processing:** The paper spends 10 pages building a complex "Teacher-Student CNN" to process raw audio signals from the machine to guess the tool wear. We skip this entirely; our environment acts as a perfect sensor that simply *knows* the exact tool wear based on the physics equations.
2. **Action Space (Tool Changes):** The paper uses a "Mixed Action Space" where the agent outputs continuous cutting parameters *plus* a discrete action (0 or 1) deciding whether to swap the tool. Our agent only outputs the continuous cutting parameters; we handle tool changes by simply resetting the environment (episode boundaries).
3. **Surface Roughness Model:** The paper trains a teacher-student network to estimate roughness from cutting data, with the empirical regression as its comparative baseline. We use the traditional empirical regression formula throughout.
4. **Gaussian vs Beta Distribution:** The paper writes a custom PPO algorithm using a Beta distribution to sample actions within absolute bounds. We use standard `Stable-Baselines3`, which uses a Gaussian distribution (clipped to bounds).

---

### 3. Dynamic Inference Features

Because the Digital Twin is fully parameterized in Python (`MillingDTEnv`), we can change physical or engineering parameters during inference (evaluation) without retraining to see how the agent adapts.

#### The Machining Task
* `kappa_total` (default `5.0` mm): The total depth of material that needs to be removed. 
* `L_cut` (default `100.0` mm): The length of the workpiece cut.
* `D` (default `50.0` mm): The tool/workpiece diameter.

#### Material & Tool Wear Physics
* `C_wear`, `alpha`, `beta`, `gamma_wear`: The Taylor-extended tool wear constants. (e.g., increasing `C_wear` simulates a softer tool).
* `k1`, `k2`, `k3`: The surface roughness regression constants (simulate different materials).

#### Engineering Constraints
* `Ra_max` (default `1.6` $\mu m$): The surface roughness limit.
* `VB_max` (default `0.3` mm): The critical tool wear limit.

*(Note: Modifying machine bounds like `n_max` or `f_max` during inference is not recommended as it breaks the neural network's `[0, 1]` observation normalization bounds).*

---

## Part 6: Original Improvement Plan


---

**Project:** Energy-Aware Deep Reinforcement Learning for Dynamic Cutting Parameter Optimization  
**Date:** October 8, 2026  
**Based on:** Code Audit (3 specialized agents), Literature Review, Online Paper Search (`paper-search` CLI)

---

> This part is the plan as it was written on October 8, 2026, kept as the design
> record. Its code snippets and its SAC claims are the ones the plan proposed.
> The paths in the snippets are the current ones; the artifacts themselves live
> under `results/`, under the names listed in
> [Part 7, Audit of the Original Plan](#part-7-audit-of-the-original-plan) and
> in `results/RESULTS.md`. The SAC claims in
> [Part 3, Literature Comparison](#part-3-literature-comparison) were withdrawn
> after the audit found that no script in this repository had ever produced
> them.

---

### Executive Summary

This plan details improvements across three critical dimensions:
1. **Environment Physics & Gymnasium API Compliance** (`MillingDT/env/milling_env.py`)
2. **Training, Experiment Tracking & Reproducibility** (`experiments/`)
3. **Baselines, Evaluation Metrics & Statistical Rigor** (`MillingDT/baselines/`, `experiments/`, `viz/`)
4. **Literature Positioning & Novelty Benchmarking**

Each improvement includes **reasoning**, **implementation details**, **code examples**, and **estimated effort**.

---

### Part 1: Environment Module (`MillingDT/env/milling_env.py`)

#### Issue 1.1: Missing `truncated` Flag in Gymnasium API

**Problem:**
- The `step()` method returns `(obs, reward, terminated, truncated, info)` but `truncated` is always `False`.
- The episode may never reach the natural termination condition (`i == N`), causing infinite training loops in certain edge cases.
- Gymnasium recommends implementing `truncated` to handle artificial episode boundaries (e.g., max-step limits).

**Reasoning:**
- Modern Gymnasium standard (0.29) requires proper handling of both `terminated` (natural end) and `truncated` (artificial timeout).
- Without a `truncated` limit, a poorly-learning agent could stall indefinitely, wasting training resources.

**Implementation:**

```python
# Current (problematic):
def step(self, action):
    # ... physics computation ...
    terminated = (self.VB >= self.VB_max) or (self.i >= self.N)
    truncated = False  # Always False - this is the issue
    return obs, reward, terminated, truncated, info

# Improved:
def __init__(self, ...):
    # ... existing code ...
    self.max_steps = 200  # Safety limit (4 batches  50 parts)
    self.step_count = 0

def step(self, action):
    self.step_count += 1
    # ... physics computation ...
    
    terminated = (self.VB >= self.VB_max) or (self.i >= self.N)
    truncated = self.step_count >= self.max_steps  # Add artificial limit
    
    return obs, reward, terminated, truncated, info

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.step_count = 0
    # ... rest of reset logic ...
    return obs, info
```

**Why This Matters:**
- Prevents infinite loops during RL training.
- Allows fine-grained episode termination control.
- Aligns with Gymnasium best practices and Stable-Baselines3 expectations.

**Effort:** 30 minutes | **Priority:** HIGH

---

#### Issue 1.2: Missing Input Action Clipping

**Problem:**
- RL agents output continuous actions in range `[-1, 1]` (normalized by Stable-Baselines3).
- These are scaled to physical machine bounds (`n  [2000, 8000]`, `f_r  [0.05, 0.35]`).
- **No explicit clipping** occurs before scaling-out-of-range inputs could cause numerical instabilities or invalid physics.

**Reasoning:**
- While Stable-Baselines3 clipping should theoretically handle this, defensive programming prevents edge-case bugs.
- Explicit clipping ensures reproducibility and robustness across different RL agents (PPO, SAC, custom algorithms).

**Implementation:**

```python
def step(self, action):
    # Current (implicit scaling):
    # n_scaled = self.n_min + (action[0] + 1) / 2 * (self.n_max - self.n_min)
    
    # Improved (explicit clipping + scaling):
    action = np.clip(action, -1.0, 1.0)  # Safety clip
    
    # Scale from [-1, 1] to [param_min, param_max]
    n_scaled = self.n_min + (action[0] + 1.0) / 2.0 * (self.n_max - self.n_min)
    f_r_scaled = self.f_r_min + (action[1] + 1.0) / 2.0 * (self.f_r_max - self.f_r_min)
    
    # Re-clip to ensure hard bounds (defense in depth)
    n_scaled = np.clip(n_scaled, self.n_min, self.n_max)
    f_r_scaled = np.clip(f_r_scaled, self.f_r_min, self.f_r_max)
    
    # ... rest of physics ...
```

**Why This Matters:**
- Prevents NaN or invalid physics calculations.
- Improves robustness when debugging or switching RL algorithms.

**Effort:** 20 minutes | **Priority:** MEDIUM

---

#### Issue 1.3: Hard Clipping in Observation Normalization Loses Gradient Information

**Problem:**
- States are normalized to `[0, 1]` with hard clipping: `obs = np.clip(obs, 0, 1)`.
- When constraints are heavily violated (e.g., `VB > VB_max` by 50%), the clipped value is still `1.0`, losing magnitude information.
- The agent receives no feedback about "how badly" the constraint was violated.

**Reasoning:**
- Soft squashing functions (e.g., tanh, sigmoid) preserve gradient magnitude while staying bounded.
- This improves credit assignment in the RL agent's policy gradient calculations.
- Recent DRL literature (Ferrisi et al. 2025, Gao et al. 2024) uses soft normalization for better convergence.

**Implementation:**

```python
# Current (hard clipping):
VB_norm = np.clip(self.VB / self.VB_max, 0, 1)

# Improved (soft squashing):
import scipy.special

def soft_normalize(value, min_val, max_val):
    """
    Use tanh squashing to keep gradients but bound to [0, 1].
    Maps extreme violations beyond [-3, 3] smoothly.
    """
    # Normalize to [-1, 1] range first
    normalized = 2 * (value - min_val) / (max_val - min_val) - 1
    # Apply tanh squashing
    return (np.tanh(normalized) + 1) / 2  # Maps to [0, 1]

# Usage in step():
VB_norm = soft_normalize(self.VB, 0, self.VB_max)
Ra_norm = soft_normalize(self.Ra, 0, self.Ra_max)
```

**Why This Matters:**
- Better gradient flow during backpropagation through the policy network.
- Agents receive signal about constraint violation magnitude, enabling more nuanced learning.
- Improves convergence speed and final policy quality.

**Effort:** 45 minutes | **Priority:** MEDIUM-HIGH

---

#### Issue 1.4: Reward Sparsity & Intermediate Feedback

**Problem:**
- Reward is computed only at episode termination or constraint violations.
- Sparse rewards make credit assignment harder; the agent may take many steps before receiving feedback.
- Literature (Zhou et al. 2023, Komatsu et al. 2023) shows intermediate rewards accelerate convergence.

**Reasoning:**
- Dense intermediate rewards (e.g., per-pass progress rewards) improve RL sample efficiency.
- Current design gives large terminal bonuses (+200 for batch completion) but no intermediate guidance.

**Implementation:**

```python
def step(self, action):
    # ... physics computation for workpiece i ...
    
    # Existing termination rewards
    r_step = w1 * MRR_hat - w2 * E_hat
    
    # NEW: Intermediate kappa progress reward
    kappa_used = sum(depths_used_so_far) / self.kappa_total
    kappa_progress_bonus = 10 * (kappa_used - self.last_kappa_progress)
    self.last_kappa_progress = kappa_used
    
    # NEW: Constraint satisfaction bonus (small, per-pass)
    constraint_bonus = 0
    if self.Ra <= self.Ra_max:
        constraint_bonus += 2  # Reward for staying in spec
    if self.VB <= 0.5 * self.VB_max:
        constraint_bonus += 1  # Reward for low tool wear
    
    r_total = r_step + kappa_progress_bonus + constraint_bonus
    
    # Large terminal bonus still applies at end
    if terminated:
        if success_completion:
            r_total += 200
        else:
            r_total -= 1000  # Catastrophic failure
    
    return obs, r_total, terminated, truncated, info
```

**Why This Matters:**
- Faster convergence (20-30% fewer training steps, based on recent papers).
- Better sample efficiency, critical for expensive simulations.
- Agent learns to maintain constraints continuously, not just at episode end.

**Effort:** 1 hour | **Priority:** HIGH

---

### Part 2: Training & Experiment Management (`experiments/`)

#### Issue 2.1: File-Based Progress Polling is Fragile

**Problem:**
- `run_experiments.py` uses `time.sleep(1.0)` polling on a progress file written by `train_rl.py`.
- Disk I/O can be slow, unreliable on network filesystems, or blocked by antivirus.
- No graceful timeout if progress file is not updated (dead worker process).

**Reasoning:**
- Multiprocessing queues or Unix domain sockets are more reliable and efficient.
- Recent Python best practices (3.11+) use `multiprocessing.Queue` for inter-process communication.

**Implementation:**

```python
# Current (problematic):
# train_rl.py writes to progress_file every 1000 steps
# run_experiments.py polls the file

# Improved: Use multiprocessing.Queue
from multiprocessing import Queue, Process
import time

def train_worker(worker_id, config, progress_queue):
    """
    Worker that trains a single model and reports progress via queue.
    """
    import stable_baselines3
    
    env = make_vec_env("MillingDTEnv-v1", n_envs=4)
    model = stable_baselines3.PPO("MlpPolicy", env, **config)
    
    for step in range(0, config['total_timesteps'], 1000):
        model.learn(total_timesteps=1000)
        
        # Send progress via queue (non-blocking)
        try:
            progress_queue.put_nowait({
                'worker_id': worker_id,
                'timesteps': step,
                'timestamp': time.time()
            })
        except:
            pass  # Queue full; skip this update
    
    model.save(f"results/run_worker_{worker_id}/models/final_model.zip")
    return model

def monitor_experiments(num_workers, config):
    """
    Main process: spawns workers and monitors progress via queue.
    """
    progress_queue = Queue(maxsize=100)
    processes = []
    
    for i in range(num_workers):
        p = Process(target=train_worker, args=(i, config, progress_queue))
        p.start()
        processes.append(p)
    
    last_update = {i: time.time() for i in range(num_workers)}
    timeout_threshold = 300  # 5 minutes
    
    while any(p.is_alive() for p in processes):
        try:
            msg = progress_queue.get(timeout=5)
            worker_id = msg['worker_id']
            last_update[worker_id] = time.time()
            print(f"Worker {worker_id}: {msg['timesteps']} steps")
        except:
            pass  # Timeout; check for dead workers
        
        # Detect stalled workers
        for i in range(num_workers):
            if processes[i].is_alive():
                elapsed = time.time() - last_update[i]
                if elapsed > timeout_threshold:
                    print(f"Worker {i} stalled; terminating")
                    processes[i].terminate()
    
    for p in processes:
        p.join()
```

**Why This Matters:**
- Eliminates disk I/O overhead and potential race conditions.
- Detects dead workers automatically.
- Cleaner, more Pythonic code following modern best practices.

**Effort:** 1.5 hours | **Priority:** MEDIUM-HIGH

---

#### Issue 2.2: Environment Registration Conflicts

**Problem:**
- Both `train_rl.py` and `evaluate.py` call `gymnasium.register()` with the same `id="MillingDTEnv-v1"`.
- If the environment definition changes, one script may load a mismatched version.
- No central source of truth for environment parameters.

**Reasoning:**
- Environment registration should happen once, in a central location (`MillingDT/env/__init__.py`).
- Avoids version drift between training and evaluation environments.
- Aligns with best practices used in modern RL frameworks (OpenAI Gym, Atari-py, etc.).

**Implementation:**

```python
# MillingDT/env/__init__.py (NEW FILE)
import gymnasium as gym
from gymnasium.envs.registration import register

# Register once
register(
    id="MillingDTEnv-v1",
    entry_point="MillingDT.env.milling_env:MillingDTEnv",
    max_episode_steps=200,
    kwargs={
        "batch_size": 50,
        "VB_max": 0.3,
        "Ra_max": 1.6,
        "kappa_total": 5.0,
        "n_min": 2000,
        "n_max": 8000,
        "f_r_min": 0.05,
        "f_r_max": 0.35,
    }
)

from env.milling_env import MillingDTEnv

__all__ = ["MillingDTEnv"]
```

```python
# train_rl.py (UPDATED)
# Remove: gymnasium.register(...)
# Add at top:
import MillingDT.env  # Triggers registration in MillingDT/env/__init__.py

def make_vec_env(env_id, n_envs=4, seed=None):
    return stable_baselines3.common.vec_env.make_vec_env(
        env_id, n_envs=n_envs, seed=seed
    )
```

```python
# evaluate.py (UPDATED)
import MillingDT.env  # Triggers registration

eval_env = gym.make("MillingDTEnv-v1")
# ... evaluation logic ...
```

**Why This Matters:**
- Single source of truth for environment configuration.
- Prevents accidental version mismatches between training and evaluation.
- Cleaner import structure.

**Effort:** 45 minutes | **Priority:** HIGH

---

#### Issue 2.3: Seed & Reproducibility Assurance

**Problem:**
- Seeds are set in multiple places (`train_rl.py`, `evaluate.py`), making it hard to verify reproducibility.
- No centralized seeding strategy; different RL runs with the same seed may not be deterministic due to thread/process ordering.

**Reasoning:**
- Recent DRL papers (Ferrisi et al. 2025, Li et al. 2024) emphasize reproducibility via strict seeding.
- Multi-threaded/process environments require explicit random state control.

**Implementation:**

```python
# MillingDT/utils/seeding.py (NEW)
import os
import numpy as np
import torch
import random

def set_seed(seed):
    """
    Set seed globally for reproducibility across all libraries.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder='big')
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

# experiments/train_rl.py (UPDATED)
from MillingDT.utils.seeding import set_seed

def train(algo="PPO", seed=42):
    seed = set_seed(seed)  # Centralized seeding
    
    env = make_vec_env("MillingDTEnv-v1", n_envs=4, seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, ...)
    model.learn(...)
    
    return model, seed

# experiments/evaluate.py (UPDATED)
def evaluate(model_path, num_episodes=100, seed=42):
    seed = set_seed(seed)
    eval_env = gym.make("MillingDTEnv-v1")
    # ... evaluation logic ...
```

**Why This Matters:**
- Guarantees reproducible results across multiple runs.
- Enables peer review and publication without "luck-based" variations.
- Aligns with recent journal requirements (Nature, IJAMT, etc.) for reproducibility.

**Effort:** 1 hour | **Priority:** HIGH

---

### Part 3: Baselines & Visualization (`MillingDT/baselines/`, `viz/`)

#### Issue 3.1: Missing Confidence Intervals in Monte Carlo Evaluation

**Problem:**
- `evaluate.py` exports mean metrics only (mean MRR, mean energy, mean Ra violations).
- No uncertainty quantification; difficult to assess statistical significance of differences between PPO and SAC.
- Recent papers (Gao et al. 2024, Kaneko et al. 2023) report 95% CI for robustness claims.

**Reasoning:**
- Monte Carlo sweep (100 scenarios) should report confidence intervals to characterize performance variance.
- Essential for peer review: reviewers want to know if observed differences are statistically significant.

**Implementation:**

```python
# experiments/evaluate.py (UPDATED)
import pandas as pd
from scipy import stats

def evaluate_with_ci(model, env, num_episodes=100, ci=0.95):
    """
    Evaluate model over Monte Carlo scenarios and compute confidence intervals.
    """
    results = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        ep_MRR, ep_Energy, ep_Ra_max, ep_VB_max = [], [], 0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_MRR.append(info.get('MRR', 0))
            ep_Energy.append(info.get('Energy', 0))
            ep_Ra_max = max(ep_Ra_max, info.get('Ra', 0))
            ep_VB_max = max(ep_VB_max, info.get('VB', 0))
        
        results.append({
            'episode': episode,
            'MRR_mean': np.mean(ep_MRR),
            'Energy_total': np.sum(ep_Energy),
            'Ra_max': ep_Ra_max,
            'VB_max': ep_VB_max,
            'Ra_violation': 1 if ep_Ra_max > 1.6 else 0,
            'VB_violation': 1 if ep_VB_max > 0.3 else 0,
        })
    
    df = pd.DataFrame(results)
    
    # Compute statistics
    stats_dict = {}
    for col in ['MRR_mean', 'Energy_total', 'Ra_max', 'VB_max']:
        mean = df[col].mean()
        std = df[col].std()
        ci_lower, ci_upper = stats.t.interval(ci, len(df)-1, loc=mean, scale=std/np.sqrt(len(df)))
        
        stats_dict[col] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }
    
    # Compute violation rates with CI
    for col in ['Ra_violation', 'VB_violation']:
        rate = df[col].mean()
        ci_lower, ci_upper = stats.binom.interval(ci, len(df), rate)
        
        stats_dict[col] = {
            'rate': rate,
            'ci_lower': ci_lower / len(df),
            'ci_upper': ci_upper / len(df),
        }
    
    return df, stats_dict

# Output format (STDOUT + CSV):
# Algorithm: PPO
# MRR:        12.5  1.2 mm/min [95% CI: 11.3-13.7]
# Energy:     456.3  45.1 kJ [95% CI: 412.1-500.5]
# Ra_violation_rate: 2.3%  1.8% [95% CI: 0.5-4.1%]
```

**Why This Matters:**
- Demonstrates statistical rigor expected by modern journals.
- Enables readers to assess whether observed differences are real or due to noise.
- Strengthens claims of "SAC outperforming PPO" with quantitative evidence.

**Effort:** 1.5 hours | **Priority:** HIGH

---

#### Issue 3.2: Code Duplication in Visualization Pipeline

**Problem:**
- `plot_eval.py` and `plot_results.py` both compute KDE plots, failure landscapes, and comparison bar charts.
- Duplicated code (100+ lines) makes maintenance hard; a bug fix must be applied twice.
- Recent maintainability audits (Ferrisi et al., 2025) recommend modularization.

**Reasoning:**
- Extract common plotting functions into a reusable module (`viz/plotting_utils.py`).
- Reduces code duplication by ~40%, improves maintainability.

**Implementation:**

```python
# viz/plotting_utils.py (NEW)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def plot_kde_distribution(data, title, xlabel, ylabel, filename=None):
    """
    Plot KDE distribution with mean/std overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(data=data, ax=ax, fill=True, alpha=0.6)
    ax.axvline(data.mean(), color='red', linestyle='--', label=f"Mean: {data.mean():.2f}")
    ax.axvline(data.median(), color='green', linestyle=':', label=f"Median: {data.median():.2f}")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_failure_landscape(df_results, x_col, y_col, title, filename=None):
    """
    Scatter plot of failure/success landscape with constraint boundaries.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate success and failure cases
    success = df_results[df_results['success'] == True]
    failure = df_results[df_results['success'] == False]
    
    ax.scatter(success[x_col], success[y_col], marker='o', s=100, 
               alpha=0.6, label='Success', color='green')
    ax.scatter(failure[x_col], failure[y_col], marker='x', s=100, 
               alpha=0.8, label='Failure', color='red')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_comparison_bar_chart(df_comparison, metrics, algorithms, filename=None):
    """
    Grouped bar chart comparing multiple algorithms across metrics.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(algorithms))
        width = 0.2
        
        for i, algo in enumerate(algorithms):
            values = df_comparison[df_comparison['algorithm'] == algo][metric].values
            ax.bar(x[i], values.mean(), width, label=algo, 
                   yerr=values.std(), capsize=5)
        
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

# viz/plot_eval.py (REFACTORED)
from plotting_utils import plot_kde_distribution, plot_failure_landscape

def main():
    df = pd.read_csv('results/run_PPO_seed42/metrics/monte_carlo.csv')
    
    # Use centralized plotting functions
    plot_kde_distribution(df['Energy_total'], 
                         "Energy Consumption Distribution",
                         "Energy (kJ)", "Density",
                         filename="results/plots/energy_kde.png")
    
    plot_failure_landscape(df, 'MRR', 'Ra_max',
                          "Failure Landscape: MRR vs Ra",
                          filename="results/plots/failure_landscape.png")
```

**Why This Matters:**
- Reduces maintenance burden (DRY principle).
- Consistent visualization style across all plots.
- Easier to update plotting aesthetics globally.

**Effort:** 2 hours | **Priority:** MEDIUM

---

#### Issue 3.3: Physics Coefficients Drift Between Environment & Baselines

**Problem:**
- `MillingDT/env/milling_env.py` defines tool wear constants (`C_wear`, `alpha`, `beta`, `gamma_wear`).
- `experiments/run_baselines.py` defines separate constants that may diverge.
- If constants are updated in one place but not the other, baseline vs. DRL comparisons become invalid.

**Reasoning:**
- Create a central `config.py` file containing all physics parameters.
- Both environment and baselines import from this single source.

**Implementation:**

```python
# MillingDT/config/physics_constants.yaml (NEW, the plan proposed a Python module)
"""
Centralized physics and engineering constants.
Based on literature calibration:
  - Tool wear: "DRL-based optimization..." J. Manuf. Processes 156
  - Surface roughness: "Machining parameter optimization..." J. Manuf. Systems 78
"""

# Tool Wear (Taylor-Extended Model)
TOOL_WEAR_CONSTANTS = {
    'C_wear': 1.0e-2,      # Wear coefficient
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
    'k_c': 2.5,      # Specific cutting energy [Wmin/mm]
}

# Machine Constraints
MACHINE_CONSTRAINTS = {
    'n_min': 2000,         # Spindle speed min [RPM]
    'n_max': 8000,         # Spindle speed max [RPM]
    'f_r_min': 0.05,       # Feed rate min [mm/r]
    'f_r_max': 0.35,       # Feed rate max [mm/r]
    'a_p': 0.5,            # Depth of cut [mm] (fixed)
}

# Engineering Constraints
ENGINEERING_CONSTRAINTS = {
    'VB_max': 0.3,         # Tool wear limit [mm]
    'Ra_max': 1.6,         # Surface roughness limit [um]
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
    'w2': 0.4,    # Energy weight (green manufacturing - tunable)
}

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
}
```

```python
# MillingDT/env/milling_env.py (UPDATED)
from MillingDT.config import TOOL_WEAR_CONSTANTS, SURFACE_ROUGHNESS_CONSTANTS, ...

class MillingDTEnv(gym.Env):
    def __init__(self, ...):
        # Use centralized constants
        self.C_wear = TOOL_WEAR_CONSTANTS['C_wear']
        self.alpha = TOOL_WEAR_CONSTANTS['alpha']
        # ... etc
        
        self.Ra_k1 = SURFACE_ROUGHNESS_CONSTANTS['k1']
        self.Ra_k2 = SURFACE_ROUGHNESS_CONSTANTS['k2']
        self.Ra_k3 = SURFACE_ROUGHNESS_CONSTANTS['k3']
```

```python
# experiments/run_baselines.py (UPDATED)
from MillingDT.config import TOOL_WEAR_CONSTANTS, SURFACE_ROUGHNESS_CONSTANTS, ...

def evaluate_nsga2_baseline(...):
    # Use same constants as environment
    C_wear = TOOL_WEAR_CONSTANTS['C_wear']
    # ... etc
```

**Why This Matters:**
- Eliminates parameter drift between environment and baselines.
- Makes it easy to run sensitivity analyses (change `C_wear`, re-run everything).
- Ensures fair comparison between DRL agents and classical baselines.

**Effort:** 1 hour | **Priority:** HIGH

---

### Part 4: Literature Positioning & Novelty

#### Issue 4.1: Lack of Explicit Energy Consumption Benchmark

**Problem:**
- Your project's **novel contribution** is the inclusion of energy consumption (`E`) as a third optimization objective.
- Recent papers (Gao et al. 2024, Ferrisi et al. 2025, Zhou et al. 2023) focus on tool life + quality but **not explicitly on energy**.
- Need explicit comparison: "DRL with energy penalty" vs. "DRL without energy penalty" (ablation study).

**Reasoning:**
- Ablation studies are standard in peer-reviewed ML research.
- Demonstrates the specific value added by the energy objective.

**Implementation:**

```python
# experiments/train_rl_ablation.py (NEW)
"""
Ablation study: Train PPO and SAC with w2 (energy weight)  {0, 0.2, 0.4, 0.6, 0.8, 1.0}
"""

import os
from MillingDT.config import REWARD_WEIGHTS

def train_with_energy_weights(energy_weights, num_seeds=3):
    """
    Train DRL agents across multiple energy weights and seeds.
    """
    results = {}
    
    for w2 in energy_weights:
        print(f"\n=== Training with w2={w2} ===")
        
        # Modify reward weight
        REWARD_WEIGHTS['w2'] = w2
        
        for seed in range(num_seeds):
            print(f"Seed {seed}...")
            
            # Train PPO
            model_ppo = train_ppo(
                total_timesteps=500_000,
                seed=seed,
                w2=w2,
            )
            model_ppo.save(f"results/run_ablation_w2_{w2}_seed{seed}/models/final_model.zip")
            
            # Train SAC
            model_sac = train_sac(
                total_timesteps=500_000,
                seed=seed,
                w2=w2,
            )
            model_sac.save(f"results/run_ablation_w2_{w2}_seed{seed}/models/final_model_sac.zip")
            
            # Evaluate both
            ppo_results = evaluate(model_ppo, num_episodes=100, seed=seed)
            sac_results = evaluate(model_sac, num_episodes=100, seed=seed)
            
            results[f"PPO_w2_{w2}_seed_{seed}"] = ppo_results
            results[f"SAC_w2_{w2}_seed_{seed}"] = sac_results
    
    # Save ablation results
    df_ablation = pd.DataFrame([
        {
            'algorithm': row.split('_')[0],
            'w2': float(row.split('_')[2]),
            'seed': int(row.split('_')[4]),
            **results[row]
        }
        for row in results.keys()
    ])
    
    df_ablation.to_csv('results/evaluation/ablation_energy_weights.csv', index=False)
    return df_ablation

# Plotting ablation results
def plot_ablation_pareto(df_ablation):
    """
    Pareto frontier: Energy vs MRR for different energy weights.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algo in ['PPO', 'SAC']:
        data = df_ablation[df_ablation['algorithm'] == algo]
        
        # Group by w2, compute mean  std
        for w2 in sorted(data['w2'].unique()):
            subset = data[data['w2'] == w2]
            
            mrr_mean = subset['MRR_mean'].mean()
            energy_mean = subset['Energy_total'].mean()
            mrr_std = subset['MRR_mean'].std()
            energy_std = subset['Energy_total'].std()
            
            ax.errorbar(energy_mean, mrr_mean, xerr=energy_std, yerr=mrr_std,
                       marker='o', label=f"{algo} w2={w2}", capsize=5)
    
    ax.set_xlabel("Total Energy (kJ)")
    ax.set_ylabel("Mean MRR (mm/min)")
    ax.set_title("Pareto Frontier: Energy vs Throughput (Ablation Study)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('results/plots/ablation_pareto_frontier.png', dpi=300)
    plt.show()
```

**Why This Matters:**
- Demonstrates that the energy objective is **not** redundant with existing objectives.
- Shows the trade-off curve (Pareto frontier) between energy and throughput.
- Strengthens your claim of "Green Manufacturing" contribution.
- Aligns with best practices in recent DRL manufacturing papers.

**Effort:** 2 hours | **Priority:** HIGH

---

#### Issue 4.2: Direct Comparison Against Recent Literature

**Problem:**
- Your paper doesn't explicitly compare against recent baselines from 2024-2025 papers (Gao et al., Ferrisi et al., Zhou et al.).
- Online search found 44 papers; need to evaluate which are most relevant and cite them properly.

**Reasoning:**
- A literature comparison table strengthens the novelty argument.
- Identifies which prior works your method outperforms and why.

**Implementation:**

```python
# [Part 3, Literature Comparison](#part-3-literature-comparison) (NEW)
# Comparison with Recent DRL Manufacturing Papers

| Aspect | Gao et al. (2024) | Ferrisi et al. (2025) | Zhou et al. (2023) | **Our Project** |
|:---|:---|:---|:---|:---|
| **Primary Objective** | Tool wear + surface quality | Tool wear prediction | Multi-task learning | Tool wear + quality + **energy** |
| **Action Space** | Continuous (n, f_r) | Continuous (n, f_r) | Continuous (n, f_r, a_p) | Continuous (n, f_r) + multi-pass a_p |
| **RL Algorithm** | PPO | DDPG | DQN + multi-task | PPO + SAC comparison |
| **Evaluation** | Standard test episodes | Case studies | Limited scenarios | 100 Monte Carlo random sweep |
| **Robustness** | Single-scenario | Single-scenario | Limited ablation | 95% CI across 100 scenarios |
| **Energy Consideration** | Implicit (via tool life) | Implicit | Not explicitly modeled | **Explicit energy term** in reward |
| **Novelty** | Tool degradation modeling | Improved tool prediction | Multi-task transfer | Green Manufacturing (energy optimization) |

**Key Differences:**
1. Explicit energy consumption modeling (novel)
2. Extensive Monte Carlo robustness validation (100+ scenarios)
3. SAC algorithm discovery of "Eco-Zone" (energy-efficient roughing strategy)
4. Open-source Gymnasium environment for reproducibility

**Validation Against Benchmarks:**
- Our DRL agents demonstrate **15-25% energy savings** vs. constant-parameter baselines
- SAC finds energy-efficient strategies that PPO alone does not discover
- Robustness maintained across 100 randomized material/tool configurations
```

**Why This Matters:**
- Clearly positions your work within the landscape of recent research.
- Helps reviewers understand the specific novel contributions.
- Guides future readers to the most relevant prior work.

**Effort:** 1.5 hours | **Priority:** MEDIUM

---

### Part 5: Implementation Roadmap & Timeline

#### Phase 1: Critical Fixes (Week 1)
| Task | Files | Effort | Priority |
|:---|:---|:---|:---|
| Add `truncated` flag | `MillingDT/env/milling_env.py` | 30 min | HIGH |
| Add action clipping | `MillingDT/env/milling_env.py` | 20 min | MEDIUM |
| Centralize environment registration | `MillingDT/env/__init__.py`, `experiments/*.py` | 45 min | HIGH |
| Replace file-polling with Queue | `experiments/run_experiments.py` | 1.5 hrs | MEDIUM-HIGH |
| Add global seeding utility | `MillingDT/utils/seeding.py` | 1 hr | HIGH |
| **Phase 1 Total** | - | **4.5 hours** | - |

#### Phase 2: Enhancement & Rigor (Week 2)
| Task | Files | Effort | Priority |
|:---|:---|:---|:---|
| Soft normalization for observations | `MillingDT/env/milling_env.py` | 45 min | MEDIUM-HIGH |
| Intermediate feedback (kappa progress) | `MillingDT/env/milling_env.py` | 1 hr | HIGH |
| Monte Carlo with 95% CI | `experiments/evaluate.py`, `viz/` | 1.5 hrs | HIGH |
| Centralize physics constants | `MillingDT/config/physics_constants.yaml` | 1 hr | HIGH |
| Energy ablation study | `experiments/train_rl_ablation.py` | 2 hrs | HIGH |
| **Phase 2 Total** | - | **6 hours** | - |

#### Phase 3: Code Quality & Documentation (Week 3)
| Task | Files | Effort | Priority |
|:---|:---|:---|:---|
| Refactor visualization code | `viz/plotting_utils.py`, `viz/*.py` | 2 hrs | MEDIUM |
| Create literature comparison doc | [Part 3, Literature Comparison](#part-3-literature-comparison) | 1.5 hrs | MEDIUM |
| Unit tests for environment | `tests/test_env.py` | 1.5 hrs | MEDIUM |
| Update README & comments | `README.md`, `*.py` | 1 hr | MEDIUM |
| **Phase 3 Total** | - | **6 hours** | - |

---

### Part 6: Testing & Validation Strategy

#### Unit Tests
```python
# tests/test_env.py
import pytest
from MillingDT.env import MillingDTEnv

def test_action_clipping():
    """Verify actions are clipped to [-1, 1] before physics."""
    env = MillingDTEnv()
    obs, _ = env.reset()
    
    # Out-of-range action
    action = np.array([1.5, -1.5])  # Outside [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that physical parameters are within bounds
    assert 2000 <= info['n'] <= 8000
    assert 0.05 <= info['f_r'] <= 0.35

def test_truncated_flag():
    """Verify truncated flag is raised after max_steps."""
    env = MillingDTEnv()
    obs, _ = env.reset()
    
    for i in range(250):  # Exceed max_steps = 200
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 200:
            assert truncated == True, "Truncated should be True after max_steps"
            break

def test_soft_normalization_gradient():
    """Verify soft normalization preserves gradient signal."""
    env = MillingDTEnv()
    obs, _ = env.reset()
    
    # Force constraint violation
    for _ in range(100):
        action = np.array([1.0, 1.0])  # Max wear stress
        obs, _, _, _, _ = env.step(action)
    
    # Observation should still contain gradient info (not clipped to 1.0)
    VB_norm = obs[0]
    assert 0.5 < VB_norm <= 1.0, "Soft normalization should preserve magnitude"

def test_physics_constants_sync():
    """Verify environment and baselines use same physics constants."""
    from MillingDT.config import TOOL_WEAR_CONSTANTS
    from experiments.run_baselines import TOOL_WEAR_CONSTANTS as BL_CONSTANTS
    
    assert TOOL_WEAR_CONSTANTS == BL_CONSTANTS, "Physics constants must match"
```

#### Integration Tests
```python
# tests/test_training.py
def test_training_convergence():
    """Verify PPO/SAC converge on training."""
    from experiments.train_rl import train
    
    model, _ = train(total_timesteps=10_000, seed=42)
    assert model is not None
    assert model.num_timesteps == 10_000

def test_evaluate_reproducibility():
    """Verify evaluation is reproducible with fixed seed."""
    from agent.evaluate import evaluate
    
    model = PPO.load("results/run_PPO_seed42/models/final_model.zip")
    env = gym.make("MillingDTEnv-v1")
    
    results1 = evaluate(model, env, num_episodes=10, seed=42)
    results2 = evaluate(model, env, num_episodes=10, seed=42)
    
    assert (results1['MRR_mean'] == results2['MRR_mean']).all()
```

---

### Part 7: Success Criteria & Validation

After implementing all improvements:

1.  **Environment API:** Passes `gymnasium.check_env()` with no warnings.
2.  **Reproducibility:** Two runs with same seed produce identical results (bit-level).
3.  **Training Stability:** Convergence curve is monotonic; no divergence or NaN.
4.  **Robustness:** Monte Carlo sweep reports 95% CI for all metrics.
5.  **Physics Consistency:** Baselines and environment use identical constants (verified by config import).
6.  **Code Quality:** No duplicate plotting code; modular and DRY.
7.  **Literature Alignment:** Explicit ablation study showing energy objective value; cited papers (Gao, Ferrisi, etc.) in comparison table.

---

### Summary of Files to Create / Modify

| File | Action | Purpose |
|:---|:---|:---|
| `MillingDT/env/milling_env.py` | Modify | Add truncated flag, soft normalization, intermediate rewards, action clipping |
| `MillingDT/env/__init__.py` | Create | Centralize environment registration |
| `MillingDT/config/physics_constants.yaml` | Create | Single source of truth for all physics parameters |
| `MillingDT/utils/seeding.py` | Create | Centralized reproducible seeding utility |
| `experiments/train_rl_ablation.py` | Create | Energy weight ablation study |
| `experiments/evaluate.py` | Modify | Add 95% CI computation |
| `experiments/run_experiments.py` | Modify | Replace file polling with multiprocessing Queue |
| `experiments/run_baselines.py` | Modify | Import constants from centralized config |
| `viz/plotting_utils.py` | Create | Refactored plotting functions (DRY principle) |
| `viz/plot_eval.py`, `viz/plot_results.py` | Modify | Use centralized plotting_utils |
| `tests/test_env.py` | Create | Unit tests for environment fixes |
| `tests/test_training.py` | Create | Integration tests for training & evaluation |
| [Part 3, Literature Comparison](#part-3-literature-comparison) | Create | Explicit comparison with recent papers |
| `README.md` | Modify | Update with ablation study and energy results |

---

### Estimated Total Effort

- **Phase 1 (Critical):** 4.5 hours
- **Phase 2 (Enhancement):** 6 hours
- **Phase 3 (Quality):** 6 hours
- **Testing & Debugging:** 4 hours
- **Total:** ~20-22 hours spread over 3 weeks

---

### Conclusion

These improvements address critical gaps identified in the code audit:
1. **Gymnasium API compliance** and robustness
2. **Reproducibility** and seeding consistency
3. **Statistical rigor** via confidence intervals
4. **Physics consistency** via centralized constants
5. **Novelty validation** through explicit energy ablation
6. **Literature positioning** via comparison tables
7. **Code quality** via modularization and DRY principles

Implementation of these changes will strengthen the project for peer review, conference submission, and publication in leading manufacturing + AI journals (J. Manuf. Processes, IJAMT, J. Manuf. Systems, IEEE Transactions on Automation Science & Engineering).

---

## Part 7: Audit of the Original Plan


---

**Plan audited:** [Part 6, Original Improvement Plan](#part-6-original-improvement-plan) (dated October 8, 2026)
**Audit date:** October 20, 2026
**Repository commit:** `9a8bf6e` ("feat: Implement CNC milling simulation engine and material removal calculator")
**Stack under audit:** Python 3.12.13, gymnasium 1.3.0, stable-baselines3 2.9.0

The findings below are the audit as it was written on October 20, 2026, against
the commit named above. The measurements it quotes that have since been redone
are marked where they appear, and the state of the project now is in
[Part 1, Environment Specification](#part-1-environment-specification) and in
`results/RESULTS.md`. The file paths are the current ones, the files having
moved since the audit; the line numbers are the ones the audit cited.

---

### 1. Verdict

Roughly 90 percent of the plan's code changes exist on disk. Three of them do not work, and the plan's headline scientific claim (SAC) has no implementation anywhere in the repository.

| Measure | Result |
|---|---|
| Plan items with code present | 9 of 9 |
| Plan items that function as specified | 6 of 9 |
| Plan success criteria met | 5 of 7 |
| Test suite | 6 passed, 0 warnings (`pytest tests/ -v`) at the time of the audit; the suite has since grown to 34 tests, all green |
| `gymnasium` environment checker | Passes clean |

The three non-functional items are not cosmetic. They are the queue heartbeat that terminates parallel training runs (Section 4.1), the Monte Carlo sweep that does not randomize what it claims to randomize (Section 4.2), and the energy ablation whose committed results show no trade-off at all (Section 4.4). A fourth issue sits outside the plan entirely but blocks reproducibility: the core environment file has never been committed to git (Section 4.5).

---

### 2. Method

Every source file listed in the plan was read in full. The following checks were executed rather than inferred:

| Check | Command | Result |
|---|---|---|
| Unit and integration tests | `python -m pytest tests/ -v` | 6 passed in 8.50s |
| Gymnasium API compliance | `gymnasium.utils.env_checker.check_env` on `MillingDTEnv` | Passes, no warnings from the environment |
| Queue heartbeat | Direct invocation of `ProgressQueueCallback._on_step` with a populated `locals` dict | No message reaches the queue |
| Config override handling | `gym.make("MillingDTEnv-v1", C_wear=0.05, k1=14.0, ...)` | All three overrides silently ignored |
| Env info payload | Inspection of the `info` dict returned by `step()` | No `T_total` key |
| Ablation policy divergence | Deterministic rollout of all 8 ablation models | 3 models identical, 5 models distinct |
| Version control coverage | `git ls-files`, `git check-ignore -v`, `git show HEAD:<path>` | `MillingDT/env/` absent from HEAD |
| SAC presence | `grep -r "SAC" --include=*.py` | 0 matches |

All numbers quoted below come from these runs or from committed CSV artifacts.

---

### 3. Item-by-item scorecard

The line numbers are the ones the audit cited, against the audited revision.
Where a defect has since been fixed, the disposition says so.

| # | Plan item | Status | Location |
|---|---|---|---|
| 1.1 | `truncated` flag | Implemented | `MillingDT/env/milling_env.py:148`, `:324` |
| 1.2 | Action clipping | Implemented | `MillingDT/env/milling_env.py:275` |
| 1.3 | Soft tanh normalization | Implemented | `MillingDT/env/milling_env.py:226-231`, used at `:257-263` |
| 1.4 | Intermediate rewards | Partial | `MillingDT/env/milling_env.py:339-365`. Kappa progress bonus, terminal bonus, and penalties are present. The per-pass "stay in spec" bonus from the plan was replaced by a penalty design, which is a reasonable deviation. |
| 2.1 | Queue-based progress polling | **Non-functional**, fixed since | Section 4.1 |
| 2.2 | Central environment registration | Implemented | `MillingDT/env/__init__.py`; `experiments/train_rl.py:45` and `experiments/evaluate.py:33` both `import MillingDT.env`. No duplicate `register()` calls remain. |
| 2.3 | Global seeding | Implemented | `MillingDT/utils/seeding.py`; used at `experiments/train_rl.py:48`, `experiments/evaluate.py:36` |
| 3.1 | Monte Carlo 95 percent CI | Partial, both defects fixed since | CI math is correct at `experiments/evaluate.py:117-158`. Two defects degraded the output (Sections 4.2 and 4.3). |
| 3.2 | Visualization DRY refactor | Implemented | `viz/plotting_utils.py`; consumed by `viz/plot_eval.py:11`, `viz/plot_results.py:12`, `viz/plot_baselines_comparison.py:20` |
| 3.3 | Centralized physics constants | Implemented | `MillingDT/config/physics_constants.yaml`; imported by `MillingDT/env/milling_env.py:24` and `experiments/run_baselines.py:30`. Wear, roughness, and energy formulas verified identical between the two consumers. |
| 4.1 | Energy weight ablation | **Degenerate output**, redone since | Section 4.4 |
| 4.2 | Literature comparison | Implemented, unsupported claims removed | Section 5 |
| Part 6 | Unit and integration tests | Implemented | `tests/test_env.py` (24 tests), `tests/test_integration.py` (5), `tests/test_progress_callback.py` (3), `tests/test_training.py` (2), 34 in all |
| Part 7 | Success criteria | 5 of 7 met at the time of the audit | Section 6 |

---

### 4. Functional defects

#### 4.1 The queue heartbeat never fires, and it kills parallel training runs (P0)

**Files:** `experiments/train_rl.py:37`, `experiments/train_rl.py:122`, `experiments/run_experiments.py:98`

`import numpy as np` sits inside the `if __name__ == "__main__":` block at `experiments/train_rl.py:121-122`. When `experiments.run_experiments` imports the module to spawn a worker, that import never executes, so `np` does not exist in the module namespace.

The progress callback at `experiments/train_rl.py:37` evaluates `np.mean(...)` while building the queue payload. The `NameError` is raised inside a `try` block whose `except` clause is a bare `except: pass`, so it is swallowed and the payload is never sent.

Probe result with a realistic `locals` payload:

```
msg after real-path locals: []
```

Consequence: `experiments/run_experiments.py:98` never receives a heartbeat, so `last_update[slot_id]` is never refreshed. After 300 seconds the stall detector terminates the worker. Every training run longer than 5 minutes is killed, and the default configuration in `experiments/run_experiments.py:129` requests 200,000 steps.

The `results/run_ablation_w2_0_2`, `w2_0_4`, `w2_0_6`, and `w2_0_8` checkpoints on disk were therefore produced by some other route, not by the script in its current state.

**Fix:** move `import numpy as np` to module scope in `experiments/train_rl.py`, and replace the bare `except: pass` with one that logs the exception.

**Fixed since the audit.** `import numpy as np` is at module scope in
`experiments/train_rl.py`, and the callback logs a dropped heartbeat through
`self._log(...)` instead of swallowing it, so the stall detector sees the
worker it is watching.

#### 4.2 The Monte Carlo sweep does not randomize the physics it claims to (P1)

**Files:** `experiments/evaluate.py:29-35`, `MillingDT/env/milling_env.py:26-63`

`generate_random_configs` draws random values for `C_wear`, `k1`, and `k2` and passes them as environment kwargs. `MillingDTEnv.__init__` accepts only `n_min`, `n_max`, `f_min`, `f_max`, `ap_min`, `ap_max`, `VB_max`, `Ra_max`, `kappa_total`, `D`, `L_cut`, `VB_0`, `w1`, and `w2`. The three physics kwargs are discarded without warning.

Probe result:

```
C_wear override honored?  env = 0.01  requested 0.05
k1 override honored?      env = 12.5     requested 14.0
kappa_total honored?      env = 12.0     requested 12.0
```

Only workpiece geometry varies across the 100 scenarios. `README.md:146` claims robustness "across randomized material/tool configurations", and `[Part 3, Literature Comparison](#part-3-literature-comparison):14` claims "high statistical robustness across randomized physical variations". Neither statement describes what the sweep exercises.

**Fix:** add `C_wear`, `k1`, `k2` (and `k3`, `alpha`, `beta`, `gamma_wear`) as overridable kwargs in `MillingDTEnv.__init__`, and have the environment use per-instance copies of the constant dicts rather than the shared module-level objects.

**Fixed since the audit.** The environment accepts those overrides now and
works on per-instance copies of the coefficient tables, and the scenarios
themselves are sampled in `MillingDT/config/scenarios.py`, so the sweep and the training
draw from the same tables. The 100 scenarios of the sweep now vary the
material and the tool as well as the geometry.

#### 4.3 `T_total` is never exported, so the cycle-time metric is dead (P1)

**Files:** `MillingDT/env/milling_env.py:218-230`, `experiments/evaluate.py:153`

The `info` dict returned by `step()` contains `pass`, `kappa_rem`, `a_p`, `f_r`, `n`, `Ra`, `MRR`, `VB`, `Energy`, `broken`, and `is_finishing`. There is no `T_total` key, although the environment maintains `self.T_total` internally.

`experiments/evaluate.py:153` reads `info.get('T_total', 0)`, which therefore returns 0 on every step. `compute_statistics` at `experiments/evaluate.py:43` includes `T_total` in its metric list and reports a confidence interval around it.

Verified against the committed artifact `results/run_PPO_seed42/metrics/monte_carlo.csv`:

```
rows: 100
T_total unique: [0] all zero: True
```

**Fix:** add `"T_total": self.T_total` to the `info` dict and re-run the sweep.

The fix was applied and the sweep was re-run. The artifact is now
`results/run_PPO_seed42/metrics/monte_carlo.csv`, and the cycle time is reported with
its confidence interval (11.43 s, 10.06 to 12.81).

#### 4.4 The committed ablation artifact is degenerate (P1)

**Files:** `experiments/train_rl_ablation.py`, `results/evaluation/ablation_results.csv`

The script trains one PPO policy per energy weight and evaluates each over 50 episodes. The committed results table holds three rows with byte-identical metrics and zero variance:

```
w1,w2,MRR_mean,MRR_std,Energy_mean,Energy_std,Time_mean,Time_std,...
1.0,0.0,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
0.5,0.5,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
0.0,1.0,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
```

A deterministic rollout of the three corresponding checkpoints explains why. All three collapse to the same saturated action pair, spindle at 8000 RPM, feed at 0.35 mm/r for the first pass then 0.05 mm/r for the second:

```
w2_0_0: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
w2_0_5: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
w2_1_0: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
```

The underlying machinery works. The four intermediate checkpoints that are on disk but absent from the CSV do show the expected trade-off, with spindle speed and feed falling as the energy weight rises:

```
w2_0_2: n=6444.5  f_r=0.2457
w2_0_4: n=6289.6  f_r=0.2832
w2_0_6: n=6024.9  f_r=0.2499
w2_0_8: n=6031.1  f_r=0.2435
```

Two further deviations from the plan:

- The plan specifies a Pareto frontier plot of MRR against Energy. `experiments/train_rl_ablation.py:139-173` now draws two line plots of Energy and Time against the weight. The committed `results/plots/ablation_pareto_frontier.png` is an orphan from an earlier implementation and is no longer regenerated by any script.
- The plan specifies three seeds and both PPO and SAC. The implementation uses one seed and PPO only.

**Fix:** re-run the ablation across the full weight grid, write the MRR against Energy Pareto frontier, and delete the orphaned PNG.

**Redone since the audit.** The ablation artifact has been regenerated. It
holds 21 rows now, 7 weights over 3 seeds, each policy pinned to its weight and
measured at that weight, and the frontier it produces is the one in section 5.2
of [Part 3, Literature Comparison](#part-3-literature-comparison). The
degenerate table and the three identical checkpoints described above are the
state of the artifact at the time of the audit, not the state of the project
now. The two deviations have partly closed: the ablation uses three seeds
across the full grid and writes the MRR against Energy frontier, and the
orphaned PNG is regenerated. The SAC arm is still absent, and the SAC claims
in [Part 3, Literature Comparison](#part-3-literature-comparison) have been
withdrawn.

#### 4.5 Version control excludes the core environment (P0, outside the plan)

**Files:** `.gitignore:58`, `.gitignore:77`

The ignore file contains `ENV/` at line 58, intended for virtual environments, and a rule for the experiment scripts at line 77. On a case-insensitive filesystem `ENV/` also matches `env/`.

```
$ git check-ignore -v MillingDT/env/milling_env.py MillingDT/env/__init__.py experiments/train_rl_ablation.py experiments/run_pipeline.py
.gitignore:58:ENV/	MillingDT/env/milling_env.py
.gitignore:58:ENV/	MillingDT/env/__init__.py
.gitignore:77:experiments/	experiments/train_rl_ablation.py
.gitignore:77:experiments/	experiments/run_pipeline.py
```

`git ls-files` lists no path under `MillingDT/env/`, and `git show HEAD:MillingDT/env/milling_env.py` reports the path as existing on disk but not in HEAD. The file that contains every Part 1 fix from the plan has never been committed. The same applies to the ablation script and the pipeline orchestrator.

A fresh clone of this repository cannot construct the environment.

**Fix:** replace `ENV/` with `env/`, `venv/`, and `.venv/`, remove the rule for the experiment scripts, then commit `MillingDT/env/`, `experiments/train_rl_ablation.py`, and `experiments/run_pipeline.py`.

**Fixed since the audit.** The ignore rules that caused this are gone, and
`MillingDT/env/`, `experiments/`, `MillingDT/utils/` and `MillingDT/config/`
are committed now, so a fresh clone constructs the environment and trains a
policy.

---

### 5. Unsupported claims

#### 5.1 The SAC results in the literature comparison do not exist (P0)

**File:** [Part 3, Literature Comparison](#part-3-literature-comparison)

The document states at line 30: "We implemented and benchmarked both PPO and SAC", and line 22 refers to a `MillingDTEnv-v2` that was never created. The benchmarking table at line 36 carries a full column headed "Our SAC (Eco-Zone)" with four quantitative values, including a 16.4 percent energy reduction and a zero percent failure rate.

A search for SAC across every Python file in the repository returns 0 matches. There is no SAC implementation. `experiments/train_rl.py:124` restricts the algorithm choice to PPO, `experiments/evaluate.py:89` evaluates PPO only, and `experiments/run_experiments.py:131` schedules PPO only. The SAC column is not a result that is hard to reproduce, it is a result that does not exist.

This violates the plan's own success criterion 7 and the "Evidence Over Claims" principle.

**Fix:** either implement SAC end to end (train, evaluate, sweep, benchmark) or remove every SAC claim from the document. The second option is the honest one if there is no time to run the experiments.

#### 5.2 README drift (P2)

**File:** `README.md`

| Claim | Actual |
|---|---|
| Line 7: spindle speed 800 to 3200 RPM | `MillingDT/config/physics_constants.yaml:31-32`: 2000 to 8000 RPM |
| Line 8: feed 0.05 to 0.25 mm/tooth | `MillingDT/config/physics_constants.yaml:33-34`: 0.05 to 0.35 mm/r |
| Line 50: two hidden layers of 64 units | `experiments/train_rl.py:70`: `net_arch=[256, 256]` |
| Lines 75-76: the genetic algorithm and the rule based baseline | Both are implemented inside `experiments/run_baselines.py` |

The reported experimental findings at `README.md:146-148` do hold against the committed Monte Carlo artifact, at the time of the audit:

```
Success mean: 1.0
Ra_violation mean: 0.0
VB_violation mean: 0.0
Broken mean: 0.0
```

Subject to the caveat in Section 4.2 that the sweep varies geometry only.

**Corrected since the audit.** The README states 2000 to 8000 RPM and 0.05 to
0.35 mm/r now, and the baselines it names are the ones that exist. The caveat
above has fallen with it: the sweep varies the material and the tool as well
as the geometry (Section 4.2). The values quoted in the probe are the ones
the sweep produced at the time of the audit, and the artifact has been
regenerated since, after the reward scale, the controller and the wear
calibration changed (Part 2, section 11). The numbers it produces now are the
ones in section 5.1 of
[Part 3, Literature Comparison](#part-3-literature-comparison).

---

### 6. Success criteria status

The statuses below are the ones the audit recorded. Three of them have been
re-measured since, and the note says which.

| # | Criterion | Status | Note |
|---|---|---|---|
| 1 | Passes `gymnasium.check_env()` with no warnings | Met | Verified against `MillingDTEnv` |
| 2 | Two runs with the same seed produce identical results | Met | `tests/test_training.py:57` passes. Bit-level identity of full PPO runs is not separately proven. |
| 3 | Convergence curve monotonic, no divergence or NaN | Met | The TensorBoard runs the audit saw are gone: the training curves are written by the CSV logger of Stable-Baselines3 into `results/run_<config>/log/progress.csv`, one per policy, and plotted by `viz/plot_results.py` |
| 4 | Monte Carlo sweep reports 95 percent CI for all metrics | Met now | `T_total` was identically zero (Section 4.3); it is exported in the info payload now, and the sweep reports all five metrics with intervals |
| 5 | Baselines and environment use identical constants | Met | Verified by reading both consumers of `MillingDT/config/physics_constants.yaml` |
| 6 | No duplicate plotting code | Met | `viz/plotting_utils.py` is the single source |
| 7 | Ablation study demonstrating the energy objective | Met now, with one exception | The output is no longer degenerate (Section 4.4): 7 weights over 3 seeds with a measured frontier. The SAC comparison is still absent (Section 5.1) |

---

### 7. Remediation order

| Order | Action | Effort | Unblocks |
|---|---|---|---|
| 1 | Fix `.gitignore` and commit `MillingDT/env/`, `experiments/train_rl_ablation.py`, `experiments/run_pipeline.py` | 10 min | Everything else. A reviewer cannot currently read the environment. |
| 2 | Move `import numpy as np` to module scope in `experiments/train_rl.py` | 5 min | Parallel multi-seed training (Section 4.1) |
| 3 | Add physics kwargs to `MillingDTEnv.__init__` and export `T_total` in `info` | 30 min | Meaningful Monte Carlo sweep (Sections 4.2, 4.3) |
| 4 | Re-run the ablation across the full weight grid, produce the MRR against Energy Pareto frontier, remove the orphaned PNG | 2 hrs (compute) | Plan items 4.1 and 7 |
| 5 | Remove the SAC claims from [Part 3, Literature Comparison](#part-3-literature-comparison), or implement SAC | 1 hr, or a full training budget | Honesty of the literature positioning |
| 6 | Correct the README parameter and structure drift | 20 min | Reader accuracy |

Steps 1 and 2 are small and gate everything downstream. Step 3 is required before the Monte Carlo numbers can support the robustness claim they are currently attached to.

Steps 1 to 4 have been carried out: the environment and the agent scripts are
committed, the heartbeat reaches the queue, the physics overrides and
`T_total` are honoured, and the ablation has been re-run across the full grid
with the Pareto frontier. Step 5 was taken as the second option, the SAC
claims were removed, and step 6 is with the README.

---

### Appendix A. Reproducing the findings

```powershell
# Test suite
python -m pytest tests/ -v

# Environment checker
python -c "import sys; sys.path.insert(0,'.'); import gymnasium as gym, MillingDT.env; \
from gymnasium.utils.env_checker import check_env; check_env(gym.make('MillingDTEnv-v1').unwrapped)"

# Section 4.1, queue heartbeat
python -c "import sys; sys.path.insert(0,'.'); import experiments.train_rl as t; \
q=[]; F=type('F',(),{'put_nowait':lambda s,x: q.append(x)}); \
cb=t.ProgressQueueCallback(F()); cb.init_callback(None); \
cb.locals={'infos':[{'reward':1.0}]}; cb._on_step(); print(q)"

# Section 4.2, ignored physics kwargs
python -c "import sys; sys.path.insert(0,'.'); import gymnasium as gym, MillingDT.env; \
u=gym.make('MillingDTEnv-v1', C_wear=0.05, k1=14.0).unwrapped; \
print(u.tool_wear_cfg['C_wear'], u.roughness_cfg['k1'])"

# Section 4.5, version control coverage
git check-ignore -v MillingDT/env/milling_env.py experiments/train_rl_ablation.py
git ls-files | Select-String 'MillingDT/env/'
```

### Appendix B. Artifacts on disk at audit time

This table is the state of the repository on October 20, 2026. The
measurements in it have since been redone: the ablation is 21 policies at
`results/run_ablation_w2_{w2}_seed{seed}`, the Monte Carlo sweep is
`results/run_PPO_seed42/metrics/monte_carlo.csv`, and the TensorBoard runs are gone,
replaced by the CSV training curves under `results/run_<config>/log/`.

| Path | Contents |
|---|---|
| `results/run_PPO_seed42/` | Primary policy plus `vecnormalize.pkl` |
| `results/run_ablation_w2_0_0/` through `w2_1_0/` | 8 ablation policies (weights 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0) |
| `results/run_PPO_seed42/metrics/monte_carlo.csv` | 100 scenario rows. Success 1.0, violations 0.0, `T_total` all zero |
| `results/evaluation/ablation_results.csv` | 3 rows, degenerate (Section 4.4) |
| `results/plots/ablation_pareto_frontier.png` | Orphan, not regenerated by any current script |
| `results/tensorboard/PPO_v4_seed42_1` to `_16` | 16 training runs, since replaced by the CSV training curves |
---

## Document map

| Where to look | What is there |
| :--- | :--- |
| `README.md` | Overview, the method in brief, the execution guide, and the measured findings |
| `PROJECT.md` | This file: every plan and reference document |
| `results/RESULTS.md` | The result tables, generated from the artifacts |
| `docs/papers/` | The three papers this work is compared against |
| `docs/Gym_Env_and_Project_Plan_Simple.pdf` | The original project proposal |
| `IMPROVEMENT_PLAN.md`, `IMPROVEMENT_PLAN_AUDIT.md` | Consolidated into Part 6 and Part 7 of this file |
