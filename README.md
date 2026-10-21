# Digital Twin-Driven DRL for Energy-Aware CNC Milling Parameter Optimization

**Course:** ME6324 - AI in Manufacturing (IIT Madras)

## Overview
This project applies Deep Reinforcement Learning (DRL) to address the dynamic cutting parameter optimization problem in CNC face milling operations. Utilizing a high-fidelity Python-based Digital Twin grounded in empirical manufacturing physics, an autonomous agent dynamically optimizes cutting parameters across multi-pass machining:
- **Spindle Speed ($n$):** 2000 - 8000 RPM
- **Feed per Revolution ($f_r$):** 0.05 - 0.35 mm/r
- **Axial Depth of Cut ($a_p$):** 0.1 - 2.5 mm

The optimization framework addresses a constrained multi-objective problem:
1. **Productivity Maximization:** Maximize Material Removal Rate (MRR).
2. **Energy Efficiency:** Minimize total electrical energy consumption ($E_{\text{total}}$), accounting for both machine idle power and cutting power.
3. **Surface Quality Assurance:** Strictly adhere to finished workpiece roughness requirements ($Ra \le 1.6\ \mu\text{m}$).
4. **Tool Wear and Breakage Avoidance:** Limit cumulative flank wear ($VB \le 0.3\ \text{mm}$) and mitigate catastrophic tool failure risks.

---

## Digital Twin Physics Formulation

The digital twin models empirical interactions derived from machining literature:
- **Material Removal Rate:**
  $$MRR = v_c \cdot a_p \cdot f_r$$
  where cutting velocity is $v_c = \frac{\pi D n}{1000}$ (m/min), with cutter diameter $D = 50\ \text{mm}$.
- **Flank Wear Rate (Extended Taylor Model):**
  $$\Delta VB = C_{\text{wear}} \cdot v_c^{\alpha} \cdot f_r^{\beta} \cdot a_p^{\gamma_{\text{wear}}} \cdot t_{\text{cut}}$$
- **Surface Roughness Model:**
  $$Ra = k_1 f_r^2 + k_2 VB + \frac{k_3}{v_c}$$
- **Energy Consumption:**
  $$k_c(v_c) = k_{c0} \left(\frac{v_c}{v_{\text{ref}}}\right)^{m}, \qquad P_{\text{total}} = P_{\text{idle}} + k_c(v_c) \cdot MRR$$
  $$E_{\text{cut}} = P_{\text{total}} \cdot t_{\text{cut}}$$
  The specific cutting energy grows with the cutting speed, so the energy of a finished part is $E_{\text{part}} = k_c(v_c) V + P_{\text{idle}} T_{\text{part}}$: the cutting term rises as the spindle speeds up and the idle term falls as the cycle time shortens. The part energy therefore has an interior minimum, which is what makes the energy objective a trade-off against the removal rate rather than a duplicate of it.

### Calibrated Constants
All values live in `MillingDT/config/physics_constants.yaml` and are shared by the environment, the baselines, the controller and the ablation study, so no consumer can drift from another.

| Group | Coefficients | Values |
| :--- | :--- | :--- |
| Tool wear | $C_{\text{wear}}, \alpha, \beta, \gamma_{\text{wear}}$ | 1.0e-2, 0.3, 0.2, 0.15 |
| Surface roughness | $k_1, k_2, k_3$ | 12.5, 0.8, 0.5 |
| Energy | $P_{\text{idle}}, k_{c0}, m$ | 500 W, 2.5 W$\cdot$min/mm$^3$, 0.6 |
| Machine | $n$, $f_r$, $a_p$ bounds | 2000-8000 RPM, 0.05-0.35 mm/r, 0.1-2.5 mm |
| Engineering | $VB_{\max}$, $Ra_{\max}$ | 0.3 mm, 1.6 $\mu$m |
| Workpiece | $D$, $L_{\text{cut}}$, $\kappa$ | 50 mm, 100 mm, 5 mm |
| Reference condition | $n$, $f_r$, $a_p$ | 5000 RPM, 0.2 mm/r, 2.5 mm |

#### The Wear Calibration
The wear coefficient is anchored on the tool life, not on a wear equation taken
from a paper: none of the three reference papers in `docs/papers/` carries a
Taylor wear model, so there was never an equation to calibrate the coefficient
against. It is anchored on the tool life that Wang et al. report (Journal of
Manufacturing Systems 78, 2025): 40 parts cut with 5 identical tools, about 8
parts per tool. $C_{\text{wear}} = 1.0\times10^{-2}$ gives about 9 parts per
tool at the reference cutting condition, so the wear constraint binds inside a
part and the tool life is a resource a batch can run out of. The previous value
of $1.1\times10^{-4}$ gave a tool life of about 870 parts, which made the wear
constraint inert inside a part and the tool life irrelevant to a batch.

The exponents are the exponents of the wear rate, not the Taylor tool life
exponents, so the tool life varies by only a factor of a few across the
machine envelope. That is a known weakness of the model and it is stated here
rather than hidden.

### Safety Factor Limits
To avoid operating at the boundary of failure, the reward formulation incorporates an engineering safety factor ($S_f = 0.85$):
- **Safety Roughness Limit:** $Ra_{\text{safe}} = 0.85 \times 1.6\ \mu\text{m} = 1.36\ \mu\text{m}$
- **Safety Tool Wear Limit:** $VB_{\text{safe}} = 0.85 \times 0.3\ \text{mm} = 0.255\ \text{mm}$

---

## Reinforcement Learning Architecture

The control policy is trained using **Proximal Policy Optimization (PPO)**:
- **State Representation (11-dim):**
  $$s_t = \left[ \widetilde{VB}, \widetilde{\kappa}, \tilde{n}_{\text{prev}}, \tilde{f}_{\text{prev}}, \tilde{a}_{p,\text{prev}}, \widetilde{MRR}_{\text{prev}}, \widetilde{Ra}_{\text{prev}}, \mathbb{I}_{\text{broken}}, w_2, \widetilde{C_{\text{wear}}}, \widetilde{k_1} \right]$$
  The eight physics features are normalized with soft tanh-squashing, so the gradient magnitude survives a constraint violation. The three context features are always present: the energy weight the part is optimized for, and the material the part is cut from. A task with a pinned weight is the same environment with a constant there, so every policy works on the same observation shape.
- **Action Space (3-dim continuous):**
  $$a_t = [a_n, a_f, a_p] \in [-1, 1]^3 \mapsto [n, f_r, a_p]$$
  The policy output is unbounded and is clipped to this box before the physical parameters are computed, which is what keeps the cut inside the machine envelope.
- **Network Architecture:**
  - Multi-Layer Perceptron (Actor-Critic `MlpPolicy`)
  - Two hidden layers of 256 units each with Tanh activations for both policy ($\pi_\theta$) and value function ($V_\phi$).
- **Reward Formulation:**
  $$r_t = 100 \left( w_1 \frac{MRR}{MRR_{\text{max}}} - w_2 \frac{E}{E_{\text{ref}}} \right) + r_{\text{progress}} + r_{VB} + r_B + r_{Ra} + r_{\text{step}} + r_{\text{HV}}$$
  - Throughput vs. Energy Trade-off: Controlled by weights $w_1$ and $w_2$, with $w_1 + w_2 = 1$.
  - $MRR_{\text{max}}$ is the largest material removal rate reachable on the machine, and $E_{\text{ref}}$ is the energy of a pass cut under the reference condition (5000 RPM, 0.2 mm/r, 2.5 mm). A pass under that condition normalizes to 1, which puts the energy term at the same sensitivity as the throughput term. With the energy scaled against a whole part instead, the shaping terms decide the policy on their own and the weight $w_2$ changes nothing.
  - The scale of the objective term is what makes the throughput count. At 5 the whole throughput and energy trade-off of a part was worth about 1.2 points of reward against a completion bonus of 200, so a policy that finished every part at any cost scored well. At 100 the trade-off is worth about fifty points per part, a quarter of the value of a part, and the completion bonus still dominates the decision to finish. The completion bonus is the value of a part, and the objective term is set against it.
  - Every one of these magnitudes lives in the `reward` section of `MillingDT/config/physics_constants.yaml`. The scale is a stated design decision and not a number buried in the environment code, and the environment, the controller and the baselines all read it from there.
  - Progress Bonus: $r_{\text{progress}} = 20 \times \Delta \text{Stock Removed Ratio}$, the anti-stall shaping.
  - Constraint Penalties:
    - $r_{VB} = -50.0$ if $VB > VB_{\text{safe}}$
    - $r_B = -100.0$ if catastrophic tool breakage occurs
    - $r_{Ra} = -50.0$ if the final pass ends outside $Ra_{\text{safe}}$
    - Step penalty: $r_{\text{step}} = -0.05$ per pass
  - Success Bonus: $+200.0$ for completing the stock within spec, plus a Pareto margin bonus of at most $+5.0$ for the margin left below the roughness limit and below the reference cycle time.

#### The Reward Is Not Neutral

The reward credits the removal rate of every pass. That is the standard way to
express a rate in a reward that is paid per pass, but the total credit then
grows with the number of passes a plan takes, so a plan that cuts the same
part in more, shallower passes is paid more for it. This is a property of the
reward, not of either method, and it has to be measured rather than argued
about.

Measured on the same 100 randomized scenarios at w2 = 0.4, by
`experiments/reward_sensitivity.py`:

| | The controller | The policy |
| :--- | ---: | ---: |
| Passes per part | 5.85 | 4.00 |
| Objective credit collected | +0.413 | +0.326 |
| Mean reward | 248.9 | 233.6 |

The two plans remove the same stock allowance. The controller takes more
passes, so it collects 27% more objective credit for the same material, and
at the objective scale of the configuration that is about nine of the fifteen
reward points that separate the two arms in the ideal case. The ideal case
comparison is therefore biased against the learned policy, and the defensible
reading of it is that the two methods are comparable, with the policy ahead
on the physical metrics and behind on the tool breakage.

Three things were checked:

1. **The objective scale is not the lever.** The controller wins at every
   scale from 5 to 4000, and the gap only grows with it, from 7.1 points at 5
   to 352 points at 4000. The sign never flips. The table is in
   `results/evaluation/reward_sensitivity.csv`.
2. **The part level alternative was implemented and rejected.** Crediting the
   material of the part against the energy of the part removes the pass count
   from the accounting, and under it the gap at the current scale is 5.0
   points instead of 15.3, and the policy is ahead at the two largest scales.
   It is not the right fix, because the material a part gives up is fixed by
   its stock: the volume of a part is `pi * D * kappa * L_cut / 1000`,
   independent of every cutting parameter. The throughput weight would then be
   inert, the reward would be an energy objective with a throughput weight
   that does nothing, and the weight sweep would stop measuring what it
   claims to measure.
3. **What a correct formulation would need.** The removal rate of a part is
   the ratio of a fixed volume to the part's time, and a rate cannot be
   accumulated over a part without becoming a volume. A correct reward has to
   price the cycle time directly, or compare the part's average rate to a
   reference at the end of the part. Both change the meaning of the weights,
   and the second makes the objective sparse and the credit assignment poor.
   Neither was adopted, and the reward as shipped carries the artifact
   instead, documented rather than hidden.

---

## Method

### One Policy, One Observation
There is a single registration for the part, `MillingDTEnv-v1`, and a single
trained policy. The batch environment is the one other registration, and it is
described below. The observation carries the energy weight the part is
optimized for and the material it is cut from, next to the eight physics
features. Both the weight and the scenario are drawn per episode during
training, so one policy covers the whole weight range and the whole material
distribution. At runtime the weight is a query, not a retrain:
`python -m experiments.evaluate --sweep-weight` walks the frontier of that one
policy, and `--compare-only` puts it against the 21 policies of the ablation study.

### Domain Randomization
`MillingDTEnv(randomize=True)` draws a new machining scenario for every part: the stock allowance, the tool diameter, the cutting length, the wear and roughness coefficients, and the initial flank wear of the tool. The bands live in `MillingDT/config/physics_constants.yaml` and are shared with the Monte Carlo sweep, so the policy is trained on the distribution it is measured on. `randomize_weight=True` draws the energy weight as well, which is what makes one policy cover the whole trade-off.

### One Physical Model
`MillingDT/config/physics.py` holds the equations of the digital twin: the cutting speed, the specific cutting energy, the material removal rate, the flank wear, the surface roughness and the energy of a pass. The environment, the classical baselines and the dashboard all import them, so a scenario that overrides a coefficient gets the physics of that scenario everywhere and no consumer can drift from another.

### Receding-Horizon Controller
`MillingDT/baselines/receding_horizon.py` is a part-level maximizer of the
expected episode reward. At every pass it simulates the remaining passes of the
part for every candidate on a grid of the machine envelope, weights the terms of
a pass by the probability that the tool survives to it, prices the breakage
risk at every pass where the tool can break, and adds the completion bonus and
the Pareto margin weighted by the probability that the tool survives all of
them. It is a stronger classical baseline than the static NSGA-II solution:
three of the four static baselines finish outside the roughness safety limit
(1.36 um) at 1.56 um, and the fourth, the constant-safe policy, finishes inside
it at 0.26 um but takes 768 s and 386 kJ per part. The controller finishes at
1.25 um and is the strongest baseline in the repository.

The plan of a part is not one parameter set but two: the roughing passes, all
but the last, run on one feed, and the finishing pass, the one that removes the
last of the stock, runs on another. The reward penalizes the roughness of the
finishing pass only, so a plan that holds one feed for the whole part is forced
to satisfy the roughness limit on the worst pass of the part and throws away
the throughput a roughing feed would have given. A controller that re-plans at
every pass is not obliged to commit to one feed for the rest of the part, so it
plans the pair.

### Transfer to a New Material
`experiments/transfer_experiment.py` measures how much of a new material a policy that was trained on another one already knows. The same target task (a harder alloy, a faster wearing tool) is evaluated four ways: the source policy zero shot, the source policy fine tuned, a policy trained on the target from scratch for the same budget, and the production policy, which was trained across a distribution of materials and needs no adaptation at all.

### The Robustness Experiment
`experiments/robustness_experiment.py` puts the learned policy against the
controller in the three regimes where the comparison decides something: the
nominal machine, where the controller solves with the exact model, the exact
state and the material in its hand; a machine that cuts a material the
controller has no model for; and a machine whose wear measurement is noisy by
5%. The artifact is `results/evaluation/robustness_results.csv`, and the
result is finding 1.

### The Batch Environment
`MillingDT/env/milling_batch_env.py` holds `MillingDTBatchEnv`, registered as
`MillingDTBatchEnv-v1`. One episode is a batch of ten parts cut with one tool.
The observation is the eleven features of the part plus the tool life left and
the fraction of the batch still to cut. The flank wear carries from one part to
the next, the tool is replaced when it breaks or reaches its wear limit, and
the replacement costs 500 reward units against the 200 a finished part is worth.
The experiment is `experiments/batch_experiment.py`, and the result is finding 7.

### Starting the Policy from the Controller
`experiments/controller_experiment.py` asks whether a policy whose actor is
initialized from the receding-horizon controller ends up better than the expert
it started from. The machinery is in `MillingDT/agent/controller_init.py`. The
controller's behaviour is recorded on 2000 randomized episodes, as
(observation, action) pairs. A supervised actor with the same shape as the PPO
actor is fitted on those trajectories, and the fitted weights are copied into a
fresh PPO policy before the reinforcement learning starts. The intent is that
the learning starts from a feasible behaviour instead of from noise. Under the
current reward the initialization is actively harmful, and finding 8 gives the
measurement and the reason.

---

## Repository Structure

```text
.
|-- MillingDT/                     # the library package
|   |-- __init__.py                # exports ENV_ID, MillingDTEnv, ALL_CONSTANTS
|   |-- agent/
|   |   |-- controller_init.py     # the ClonedActor and the controller dataset
|   |-- baselines/
|   |   |-- receding_horizon.py    # the controller that re-solves at every pass
|   |-- config/
|   |   |-- __init__.py            # loads the two YAML files, exposes the constants and the paths
|   |   |-- physics_constants.yaml # the calibrated constants, the reward weights, the scenario tables
|   |   |-- paths.yaml             # where the project keeps its runs, logs, plots and tables
|   |   |-- physics.py             # the physical model
|   |   |-- scenarios.py           # the randomized machining scenarios
|   |-- env/
|   |   |-- __init__.py            # the single registration point
|   |   |-- milling_env.py         # the Gymnasium environment
|   |   |-- milling_batch_env.py   # the batch environment, one tool cuts a batch
|   |-- sim/machining_sim.py       # the trajectory simulation behind the dashboard
|   |-- utils/seeding.py           # centralized reproducible seeding
|-- experiments/                   # the entry points
|   |-- batch_experiment.py        # the batch against the part
|   |-- controller_experiment.py   # the controller initialization experiment
|   |-- evaluate.py                # the Monte Carlo sweep and the runtime weight sweep
|   |-- reward_sensitivity.py     # how much of the comparison against the controller the reward decides
|   |-- robustness_experiment.py   # the controller against the policy in three regimes
|   |-- run_baselines.py           # the five classical baselines
|   |-- run_experiments.py         # the parallel supervisor for multi-seed runs
|   |-- run_pipeline.py            # the master execution script
|   |-- summarize_results.py       # builds the result tables quoted in the docs
|   |-- train_rl.py                # the one training entry point
|   |-- train_rl_ablation.py       # the energy weight ablation
|   |-- transfer_experiment.py     # transfer to a new material
|-- viz/                           # the figures
|   |-- plot_results.py            # the training curves from the progress.csv files
|   |-- plot_eval.py               # KDE distributions and failure landscape plots
|   |-- plot_baselines_comparison.py # the comparison against the classical baselines
|   |-- plot_controller_comparison.py # the controller against the two learned policies
|   |-- plotting_utils.py          # shared plotting helpers, single source for figure style
|-- results/                       # every run, one directory per training configuration
|   |-- run_PPO_seed42/            # the production policy
|   |-- run_PPO_batch_seed42/      # the batch-aware policy
|   |-- run_PPO_noisystate_seed42/ # the policy trained against a noisy wear measurement
|   |-- run_<config>/
|   |   |-- models/                # final_model.zip, vecnormalize.pkl
|   |   |-- log/                   # progress.csv, the training curve
|   |   |-- metrics/               # the evaluation tables of that policy
|   |   |-- plots/                 # the figures of that policy
|   |-- evaluation/                # the tables that compare several runs
|   |   |-- baselines.csv          # the five classical baselines
|   |   |-- ablation_summary.csv   # the energy weight ablation, 7 weights over 3 seeds
|   |   |-- weight_sweep_vs_ablation.csv # the one policy against the 21
|   |   |-- transfer_results.csv   # the transfer to a harder material
|   |   |-- controller_init_results.csv # the controller initialization, three arms
|   |   |-- robustness_results.csv # the controller against the policy, three regimes
|   |   |-- reward_sensitivity.csv # the reward scale and form behind the comparison against the controller
|   |   |-- batch_results.csv      # the batch against the part
|   |-- plots/                     # the cross-run figures
|   |-- RESULTS.md                 # the generated summary of every artifact
|-- app.py                         # the Streamlit interactive CNC Digital Twin workstation
|-- docs/
|   |-- papers/                     # the three papers this work is compared against
|   |-- Gym_Env_and_Project_Plan_Simple.pdf # the original project proposal
|-- PROJECT.md                     # Every plan and reference document, in one file
`-- tests/                         # Pytest test suite for physical consistency
```

---

## Execution Guide

### 1. Master Pipeline (Recommended)
To run the full workflow (production policy training, the Monte Carlo sweep, the runtime weight sweep, the energy weight ablation, the classical baselines, the transfer experiment, the controller initialization, the robustness experiment, the batch experiment, the plots and the result tables, eleven stages in all):
```bash
python -m experiments.run_pipeline --steps 1000000 --ablation_steps 150000
```

### 2. Standalone Training & Evaluation
- **Train the production policy** (randomized scenarios and a randomized energy weight, so one policy covers the whole trade-off and the whole material distribution):
  ```bash
  python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight
  ```
- **Train a policy for one point of the trade-off** (used by the ablation study):
  ```bash
  python -m experiments.train_rl --steps 150000 --seed 42 --randomize --w2 0.6
  ```
- **Train the batch-aware policy** (the batch environment, one tool cutting a batch of ten parts):
  ```bash
  python -m experiments.train_rl --env-id MillingDTBatchEnv-v1 --steps 1000000 --seed 42 --randomize --run-name PPO_batch_seed42
  ```
- **Train the policy that faces a noisy wear measurement** (5% noise on the wear and the remaining stock):
  ```bash
  python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight --obs-noise 0.05 --run-name PPO_noisystate_seed42
  ```
- **Monte Carlo Statistical Sweep:**
  ```bash
  python -m experiments.evaluate --scenarios 100
  ```
- **Runtime Weight Sweep** of the one policy, and the comparison against the per-weight policies:
  ```bash
  python -m experiments.evaluate --sweep-weight --scenarios 100
  python -m experiments.evaluate --compare-only
  ```
- **Energy Weight Ablation Study** (7 energy weights, 3 seeds each, trained in parallel):
  ```bash
  python -m experiments.train_rl_ablation --steps 150000 --scenarios 50 --slots 7
  ```
- **Classical Baselines** (including the receding-horizon controller):
  ```bash
  python -m experiments.run_baselines
  ```
- **Transfer to a new material** (zero shot, fine tuned, from scratch, production policy):
  ```bash
  python -m experiments.transfer_experiment 300000 50000
  ```
- **Start the policy from the controller** (the controller, PPO from scratch, and PPO initialized from the controller, on the same 100 randomized scenarios):
  ```bash
  python -m experiments.controller_experiment 1000000 2000
  ```
- **Robustness against the classical controller** (the controller with every advantage, with a wrong model of the material, and with a noisy wear measurement):
  ```bash
  python -m experiments.robustness_experiment
  ```
- **The batch against the part** (the batch-aware policy, the same policy without the batch context, and the per-part controller):
  ```bash
  python -m experiments.batch_experiment
  ```
- **The reward sensitivity** (how much of the comparison against the controller is decided by the reward, over a range of objective scales and two forms of the objective term):
  ```bash
  python -m experiments.reward_sensitivity
  ```
- **Rebuild the result tables** quoted in this file:
  ```bash
  python -m experiments.summarize_results
  ```

### 3. Generate Visualizations
- **Training Convergence Curves:**
  ```bash
  python -m viz.plot_results
  ```
- **KDE Constraint Distributions & Failure Landscapes:**
  ```bash
  python -m viz.plot_eval
  ```
- **Comparative Baseline Benchmarks** (the six policies side by side):
  ```bash
  python -m viz.plot_baselines_comparison
  ```
- **Controller Against the Learned Policies** (the three arms of the controller initialization experiment):
  ```bash
  python -m viz.plot_controller_comparison
  ```
All output figures are saved to `results/plots/`, except the Monte Carlo
distributions and the failure landscape, which belong to the production run and
are written to `results/run_PPO_seed42/plots/`. The index of every figure is at
the end of `results/RESULTS.md`.

### 4. Interactive Digital Twin Workstation
Launch the Streamlit web dashboard to simulate real-time cutting, inspect 2D workpiece stock depletion, view micro-topography cusps, and generate ISO 6983 CNC G-code:
```bash
streamlit run app.py
```

### 5. Compute Notes
PPO on a 256x256 MLP does not benefit from a GPU. The bottleneck is the
per-step forward pass and the Python environment step, not the gradient
math, so the measured throughput on this machine (16 cores, RTX 4060) is:

| Configuration | Throughput |
| :--- | ---: |
| CPU, 1 torch thread | 812 steps/s |
| CPU, 4 torch threads | 881 steps/s |
| CPU, 8 torch threads | 913 steps/s |
| RTX 4060, `--device cuda` | 318 steps/s |

Stable-Baselines3 warns about the same thing for a non-CNN policy. The
training scripts therefore default to the CPU with one thread per process,
and the parallelism comes from running the ablation policies as separate
processes (`experiments/run_experiments.py`, 7 slots by default), which is what
uses the cores. A GPU would pay off for a convolutional policy, a much
larger network, or an off-policy algorithm with a replay buffer.

---

## Experimental Findings

1. **Inside a perfect digital twin the two methods are comparable, and the learned policy is ahead where the controller's information advantage disappears.** The robustness experiment (`results/evaluation/robustness_results.csv`, `python -m experiments.robustness_experiment`) runs the receding-horizon controller and the learned policy over the same 100 randomized scenarios at the same energy weight w2 = 0.4, in three regimes.

   | Regime | Arm | Mean reward | Reward std | MRR (mm3/min) | Energy (kJ) | Cycle (s) | Broken | Completed |
   | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | The controller has every advantage | Receding-horizon | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 3% | 96% |
   | The controller has every advantage | PPO | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 6% | 94% |
   | The controller's material model is wrong | Receding-horizon | 63.5 | 117.1 | 1306.9 | 71.0 | 23.3 | 2% | 95% |
   | The controller's material model is wrong | PPO | 226.9 | 88.3 | 1604.2 | 66.3 | 11.5 | 6% | 94% |
   | The wear measurement is noisy | Receding-horizon | 181.3 | 185.9 | 1300.4 | 74.2 | 31.1 | 4% | 91% |
   | The wear measurement is noisy | PPO trained with the noise | 213.6 | 87.3 | 1554.2 | 66.9 | 12.0 | 6% | 94% |

   The artifact holds a third row for the model error regime, the policy that is told the material: 215.1 mean reward, 94.2 std, 1548.7 MRR, 66.8 kJ, 12.2 s, 6% broken, 94% completed. It says that the policy does not need the material to be told: the policy that believes the nominal material (226.9) is slightly better than the one that is told the truth (215.1).

   - **In the nominal regime the two methods are comparable, and the measured gap is partly the reward's artifact.** The controller plans the part with the exact model, the exact state and the material in its hand, and it scores 248.9 against the policy's 233.6. The gap is inside the noise: the standard error of the mean is about 8 for both arms, so a difference of 15 has a t of about 1.4. About nine of those fifteen points are the artifact of the reward, which credits the removal rate of every pass: the controller takes 5.85 passes against the policy's 4.00, so it collects 27% more objective credit for the same material. The comparison is biased against the learned policy.
   - **The learned policy is the better throughput operator.** It removes 22% more material per minute (1608 against 1318), finishes a part in a third less time (11.4 against 17.1 s) and spends 3.7% less energy. It pays for that with twice the tool breakage, 6% against 3%, and 4% of its finished surfaces above the roughness limit against 2%.
   - **The advantage appears where a plant actually is.** With a material the controller has no model for, its reward collapses by three quarters, from 248.9 to 63.5, because it does not slow down and its finished surfaces go out of specification. The learned policy holds at 226.9.
   - **The same holds for a noisy measurement.** With 5% noise on the wear and the remaining stock the controller's reward falls to 181.3 and its spread more than doubles, to 185.9 against the policy's 87.3. The policy trained with the same noise keeps 213.6 and cuts the part in 12.0 s against the controller's 31.1 s.

   The claim is not that reinforcement learning beats a model-based controller on the objective inside a perfect digital twin, because the controller computes the optimum of the same objective with strictly more information, and because the reward pays the controller for the passes it takes. The claim is that the learned policy is the better throughput operator, and that its advantage appears where that information advantage disappears. The scale of the reward is a design decision, and it decides how the ideal case is scored. So is the form in which the reward pays for a rate: the per-pass crediting of the removal rate is worth about nine of the fifteen points between the two arms in the nominal regime, and it is documented in the reward formulation section above.

2. **The production policy over 100 randomized scenarios.** The policy is trained across randomized machining scenarios and evaluated on 100 held-out scenarios of the same kind. It completed 94.0% of the parts, kept the finished surface inside the roughness limit in 96% of them, never exceeded the flank wear limit, and broke a tool in 6%. Material removal rate 1608.5 mm3/min, energy 66.4 kJ per part, cycle time 11.4 s. The worst pass of an episode exceeds the roughness limit in 55% of the scenarios, which is the roughing pass: the reward constrains the finishing pass, because that is the pass that sets the surface of the part. The mean operating point is 7907 RPM, 0.318 mm/r, 2.07 mm of depth of cut and 4.0 passes per part. Under a reward that counts the throughput the policy is a much more aggressive operator than it was, and it pays for that with the breakage rate and with the finished surfaces.

3. **The energy weight trades throughput for energy.** The non-dominated weights run from w2 = 0.5 (1556.6 mm3/min at 69.1 kJ) to w2 = 1.0 (918.5 mm3/min at 57.2 kJ). Across that frontier the removal rate falls 41% and the energy of the part falls 17%, and the spindle speed falls with the weight, from 7859 to 5382 RPM, which is the mechanism the energy model predicts: the specific cutting energy grows with the speed, so a faster spindle buys removal rate with energy. The three lowest weights are dominated, they spend more energy without removing more material.

4. **One policy against the 21 weight-specific policies: the earlier claim is reversed.** A single policy of 1,000,000 steps is asked for every weight at runtime, and the 21 weight-specific policies of 150,000 steps each are measured at the weight they were trained for.

   | w2 | MRR, one policy | MRR, per weight | MRR gap | Energy gap |
   | ---: | ---: | ---: | ---: | ---: |
   | 0.0 | 1475.9 | 1397.1 | +5.6% | -2.9% |
   | 0.2 | 1518.0 | 1456.9 | +4.2% | -6.1% |
   | 0.4 | 1608.5 | 1524.9 | +5.5% | -4.2% |
   | 0.5 | 1630.3 | 1556.6 | +4.7% | -4.4% |
   | 0.6 | 1540.7 | 1519.8 | +1.4% | -4.0% |
   | 0.8 | 1114.5 | 1224.4 | -9.0% | -8.0% |
   | 1.0 | 720.4 | 918.5 | -21.6% | -9.7% |

   Under the old reward the one policy beat all 21 by a wide margin. Under the current reward it is ahead by 1.4% to 5.6% over the weights from 0.0 to 0.6 and behind at 0.8 and 1.0, by 9.0% and 21.6%. The weight in the observation is a weaker signal than a dedicated policy at the two heaviest weights. The honest reading: one conditional policy covers the range and is ahead over most of it, but a policy trained for one weight is better at the extreme weights, where the trade-off is steepest. The weight remains a runtime query rather than a retrain.

5. **The classical baselines are weak, and the controller is the strongest of them.** All five are measured on the same 50 episodes of the nominal condition at w2 = 0.4.

   | Policy | Reward | MRR (mm3/min) | Energy (kJ) | Cycle (s) | Final Ra (um) |
   | :--- | ---: | ---: | ---: | ---: | ---: |
   | Constant high | 10.0 | 1099.6 | 17.8 | 4.3 | 1.56 |
   | Constant safe | -1882.9 | 1.6 | 385.7 | 768.0 | 0.26 |
   | NSGA-II static | 5.1 | 732.9 | 18.8 | 6.4 | 1.56 |
   | Receding-horizon | 253.5 | 725.5 | 19.0 | 6.7 | 1.25 |
   | Rule based | 3.3 | 719.8 | 18.8 | 6.6 | 1.56 |

   Three of the four static policies finish outside the roughness safety limit (1.36 um) at 1.56 um, and the fourth, the constant-safe policy, finishes inside it at 0.26 um but takes 768 s and 386 kJ per part. The receding-horizon controller finishes at 1.25 um and is the strongest baseline. The figure `results/plots/baseline_vs_ppo_comparison.png` puts all six policies side by side, the learned one included.

6. **Training across materials is what transfers.** On a harder alloy, measured over 30 scenarios, the production policy, which was trained across a distribution of materials with the material context in its observation, needs no adaptation at all and is the best arm.

   | Arm | Mean reward | Ra violation, worst pass | Ra violation, finished surface | Parts completed |
   | :--- | ---: | ---: | ---: | ---: |
   | Source policy, zero shot | 168.5 | 100.0% | 6.7% | 86.7% |
   | Source policy, fine tuned, 50k steps | 101.9 | 26.7% | 0.0% | 86.7% |
   | Trained on the target from scratch, 50k steps | 101.9 | 26.7% | 0.0% | 86.7% |
   | Production policy, zero shot | 190.1 | 100.0% | 10.0% | 90.0% |

   Fine tuning and training on the target from scratch both reduce the roughness violations from every scenario to a quarter of them, and both lose more than the production policy, because the harder material wears the tool twice as fast and the policy that slows down for that trades away more throughput than it gains in specification.

7. **The batch context is worth fewer tool changes and less energy, and the machine time is worth a lot.** A batch is ten parts cut with one tool, and the tool change costs 500 reward units against the 200 a finished part is worth. Three arms over 30 batches:

   | Arm | Batch reward | Std | Reward per part | Tool changes | Parts cut | Batch time (s) | Batch energy (kJ) |
   | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | PPO, batch aware | 1817.0 | 477.8 | 166.8 | 0.63 | 10 | 113.1 | 610.4 |
   | PPO, per part | 1709.9 | 366.2 | 169.4 | 0.83 | 10 | 114.7 | 674.3 |
   | Receding-horizon, per part | 1859.9 | 328.8 | 173.7 | 0.83 | 10 | 207.5 | 692.0 |

   The batch context is worth 24% fewer tool changes and 9.5% less energy at the same completion, and the batch-aware policy also collects the higher batch reward, 1817.0 against 1709.9, at a lower reward per part. The per-part controller gets the highest batch reward, but it needs 207.5 s of machine time for the batch against the batch-aware policy's 113.1 s.

8. **Starting the policy from the controller is now actively harmful.** Three arms were measured on the same 100 randomized scenarios at the same energy weight w2 = 0.4: the receding-horizon controller, the production policy trained from scratch, and a policy whose actor was initialized from the controller and then trained with PPO.

   | Arm | Mean reward | Reward std | MRR (mm3/min) | Energy (kJ) | Cycle (s) | Ra violation, finished surface | Flank wear violations | Broken | Completed |
   | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | Receding-horizon controller | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 2% | 1% | 3% | 96% |
   | PPO from scratch | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 4% | 0% | 6% | 94% |
   | PPO initialized from the controller | 119.8 | 178.7 | 1274.9 | 81.5 | 42.6 | 4% | 9% | 4% | 87% |

   The reason is instructive. The cloned actor starts the policy at the controller's operating point, which is the conservative one: a low feed and a low spindle speed so that every pass of the part stays inside the roughness safety limit. With the wear constraint binding, that conservatism is expensive. The policy then learns from a bad starting point, slows down further to stay away from the wear limit, and ends up cutting a part in 42.6 s against the controller's 17.1 s and the from-scratch policy's 11.4 s. It also exceeds the flank wear limit on 9% of the parts, which the other two arms never do. The earlier revision, under the old reward, found the initialization harmless: that was the reward not counting the cost of the slow operating point. The method to keep from the controller initialization is the record and clone machinery, not the initialization itself. The figure `results/plots/controller_vs_ppo.png` shows the metrics of the three arms.

Every number above is generated from the artifacts in `results/` by
`python -m experiments.summarize_results`, which writes the same tables to
`results/RESULTS.md`. The methods behind each result are described in Part 2
of `PROJECT.md`, the position against the published work is in Part 3, and the
environment contract is in Part 1.

---

## Future Work

1. **Publish the reward sensitivity alongside the comparison against the
   controller.** The comparison against a model-based controller is sensitive
   to how the reward pays for a rate: the reward credits the removal rate of
   every pass, so a plan that cuts the same part in more, shallower passes
   collects more credit, and the ideal case comparison is biased against the
   learned policy by about nine of the fifteen points that separate the two
   arms there. The sensitivity has to be published alongside the result, and
   the measurement is `results/evaluation/reward_sensitivity.csv`, written by
   `python -m experiments.reward_sensitivity`. What a correct formulation
   would need is set out in the reward formulation section above: a reward
   that prices the cycle time directly, or that compares the part's average
   rate to a reference at the end of the part. Both change the meaning of the
   weights, and the second makes the objective sparse and the credit
   assignment poor.

---

## References

1. X. Mo, M. Xia, S. Mei, X. Hu, "Deep reinforcement learning-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", *Journal of Manufacturing Processes*, 156 (2025) 1006-1031.
2. P. Wang, Y. Cui, H. Tao, X. Xu, S. Yang, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", *Journal of Manufacturing Systems*, 78 (2025) 124-152.
3. P. Wang, Y. You, W. Liu, D. Ding, F. Xu, "Knowledge-guided deep reinforcement learning framework for feature machining step sequencing", *Int. J. Adv. Manuf. Technol.*, 140 (2025) 5617-5636.
