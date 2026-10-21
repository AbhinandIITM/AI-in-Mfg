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
All values live in `MillingDT/config/physics_constants.yaml` and are shared by the environment, the baselines and the ablation study, so no consumer can drift from another.

| Group | Coefficients | Values |
| :--- | :--- | :--- |
| Tool wear | $C_{\text{wear}}, \alpha, \beta, \gamma_{\text{wear}}$ | 1.1e-4, 0.3, 0.2, 0.15 |
| Surface roughness | $k_1, k_2, k_3$ | 12.5, 0.8, 0.5 |
| Energy | $P_{\text{idle}}, k_{c0}, m$ | 500 W, 2.5 W$\cdot$min/mm$^3$, 0.6 |
| Machine | $n$, $f_r$, $a_p$ bounds | 2000-8000 RPM, 0.05-0.35 mm/r, 0.1-2.5 mm |
| Engineering | $VB_{\max}$, $Ra_{\max}$ | 0.3 mm, 1.6 $\mu$m |
| Workpiece | $D$, $L_{\text{cut}}$, $\kappa$ | 50 mm, 100 mm, 5 mm |
| Reference condition | $n$, $f_r$, $a_p$ | 5000 RPM, 0.2 mm/r, 2.5 mm |

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
  $$r_t = 5 \left( w_1 \frac{MRR}{MRR_{\text{max}}} - w_2 \frac{E}{E_{\text{ref}}} \right) + r_{\text{progress}} + r_{VB} + r_B + r_{Ra} + r_{\text{step}} + r_{\text{HV}}$$
  - Throughput vs. Energy Trade-off: Controlled by weights $w_1$ and $w_2$, with $w_1 + w_2 = 1$.
  - $MRR_{\text{max}}$ is the largest material removal rate reachable on the machine, and $E_{\text{ref}}$ is the energy of a pass cut under the reference condition (5000 RPM, 0.2 mm/r, 2.5 mm). A pass under that condition normalizes to 1, which puts the energy term at the same sensitivity as the throughput term. With the energy scaled against a whole part instead, the shaping terms decide the policy on their own and the weight $w_2$ changes nothing.
  - Progress Bonus: $r_{\text{progress}} = 2 \times \Delta \text{Stock Removed Ratio}$, the anti-stall shaping.
  - Constraint Penalties:
    - $r_{VB} = -50.0$ if $VB > VB_{\text{safe}}$
    - $r_B = -100.0$ if catastrophic tool breakage occurs
    - $r_{Ra} = -50.0$ if the final pass ends outside $Ra_{\text{safe}}$
    - Step penalty: $r_{\text{step}} = -0.05$ per pass
  - Success Bonus: $+200.0$ for completing the stock within spec, plus a Pareto margin bonus of at most $+5.0$ for the margin left below the roughness limit and below the reference cycle time.

---

## Method

### One Policy, One Observation
There is a single environment registration, `MillingDTEnv-v1`, and a single
trained policy. The observation carries the energy weight the part is
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
`MillingDT/baselines/receding_horizon.py` re-solves the next pass at every step, with the remaining stock and the current flank wear as the state, and keeps the candidate that scores best and stays inside the roughness and wear limits. It is a stronger classical baseline than the static NSGA-II solution: the four static baselines all end outside the roughness safety limit, and this one is the only classical policy that stays inside it.

### Transfer to a New Material
`experiments/transfer_experiment.py` measures how much of a new material a policy that was trained on another one already knows. The same target task (a harder alloy, a faster wearing tool) is evaluated four ways: the source policy zero shot, the source policy fine tuned, a policy trained on the target from scratch for the same budget, and the production policy, which was trained across a distribution of materials and needs no adaptation at all.

### Starting the Policy from the Controller
`experiments/controller_experiment.py` asks whether a policy whose actor is
initialized from the receding-horizon controller ends up better than the expert
it started from. The machinery is in `MillingDT/agent/controller_init.py`. The
controller's behaviour is recorded on 2000 randomized episodes, as
(observation, action) pairs. A supervised actor with the same shape as the PPO
actor is fitted on those trajectories, and the fitted weights are copied into a
fresh PPO policy before the reinforcement learning starts. The intent is that
the learning starts from a feasible behaviour instead of from noise. The
measured outcome is in finding 6.

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
|   |-- sim/machining_sim.py       # the trajectory simulation behind the dashboard
|   |-- utils/seeding.py           # centralized reproducible seeding
|-- experiments/                   # the entry points
|   |-- controller_experiment.py   # the controller initialization experiment
|   |-- evaluate.py                # the Monte Carlo sweep and the runtime weight sweep
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
|   |-- run_<config>/
|   |   |-- models/                # final_model.zip, vecnormalize.pkl
|   |   |-- log/                   # progress.csv, the training curve
|   |   |-- metrics/               # the evaluation tables of that policy
|   |   |-- plots/                 # the figures of that policy
|   |-- evaluation/                # the tables that compare several runs
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
To run the full workflow (production policy training, the Monte Carlo sweep, the runtime weight sweep, the energy weight ablation, the classical baselines, the plots and the result tables):
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

1. **Robustness over 100 randomized scenarios.** The policy is trained across randomized machining scenarios and evaluated on 100 held-out scenarios of the same kind. It completed 99.0% of the parts, kept the finished surface inside the roughness limit in **100%** of them, never exceeded the flank wear limit, and broke a tool in 1%. Material removal rate 1205.2 mm3/min [95% CI 1088.9-1321.4], energy 68.4 kJ per part, cycle time 20.3 s. The worst pass of an episode exceeds the roughness limit in 38% of the scenarios, which is the roughing pass: the reward constrains the finishing pass, because that is the pass that sets the surface of the part.

2. **The energy weight trades throughput for energy.** The non-dominated weights run from w2 = 0.4 (1038 mm3/min at 72.7 kJ) to w2 = 1.0 (951 mm3/min at 65.8 kJ). Across that frontier the removal rate falls 8.4% while the energy of the part falls 9.5%, and the spindle speed falls from 7256 to 6490 RPM, which is the mechanism the energy model predicts: the specific cutting energy grows with the speed, so a faster spindle buys removal rate with energy. The trade-off is symmetric, which is what a real Pareto frontier looks like. The lowest weights are dominated, they spend more energy without removing more material.

3. **One policy replaces 21 weight-specific policies.** A single policy of 1,000,000 steps beats all 21 weight-specific policies of 150,000 steps each, which is 3,150,000 steps in total: 14% to 28% more material removal rate for 3% to 16% less energy across the weights, and 15% to 16% more removal rate for at most 2% more energy at the two heaviest weights. Its own frontier is monotone, 1213.6 down to 1096.0 mm3/min against 69.6 down to 67.1 kJ. The weight is a runtime query rather than a retrain.

4. **The classical baselines were weak until the controller was made dynamic.** Three of the four static policies, the NSGA-II solution, the constant-high policy and the rule-based policy, finish at 1.557 um of roughness, above the 1.36 um safety limit, and score around -46. The fourth, the constant-safe policy, finishes inside the limit at 0.068 um but takes 3000 s and 1507 kJ per part, which is what its -139 measures. The receding-horizon controller, which re-solves at every pass, finishes at 1.246 um, inside the safety limit, scores +203.5 and is the strongest baseline in the repository. The figure `results/plots/baseline_vs_ppo_comparison.png` puts all six policies side by side: every static policy is infeasible on one axis or the other, so the comparison that carries information is the controller against the learned policy.

5. **Training across materials is what transfers; fine tuning is not.** On a harder alloy, a policy trained on the nominal material scores 182.5 and violates the roughness limit in every scenario. Fine tuning it for 50,000 steps moves the score to 196.6 and removes every violation, and so does training on the target from scratch for the same budget. The production policy, which was trained across a distribution of materials with the material context in its observation, needs no adaptation at all and scores 200.3 with the finished surface in spec in every scenario. Its 60% worst-pass violations sit on the roughing passes, which the reward does not constrain.

6. **Starting the policy from the controller does not beat the controller.** Three arms were measured on the same 100 randomized scenarios at the same energy weight w2 = 0.4: the receding-horizon controller, the production policy trained from scratch, and a policy whose actor was initialized from the controller and then trained with PPO. The initialization does not produce a policy that beats the expert it started from. The controller still removes 38% more material per minute (1560.6 against 1126.1 mm3/min) and finishes a part in 13.5 s against 32.3 s. What it does produce is a far more consistent policy: the spread of the episode reward across the 100 scenarios falls from 30.0 to 3.4, an order of magnitude, and the tool breakage falls to zero. The consistency is paid for in throughput and energy, 6.6% less material removal rate and 12.9% more energy per part than the from-scratch policy. The mean reward difference against the from-scratch policy, 199.35 against 198.57, is inside the noise: the standard error of the mean over 100 scenarios is 30.0/sqrt(100) = 3.0 for the from-scratch arm, so a gap of 0.8 is not a result. The controller is the least reliable of the three arms, with a reward standard deviation of 65.9, 5% tool breakage and 95% of the parts completed. The figure `results/plots/controller_vs_ppo.png` shows all six metrics of the three arms.

Every number below is generated from the artifacts in `results/` by
`python -m experiments.summarize_results`, which writes the same tables to
`results/RESULTS.md`. The methods behind each result are described in Part 2
of `PROJECT.md`, the position against the published work is in Part 3, and the
environment contract is in Part 1.

---

## References

1. X. Mo, M. Xia, S. Mei, X. Hu, "Deep reinforcement learning-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", *Journal of Manufacturing Processes*, 156 (2025) 1006-1031.
2. P. Wang, Y. Cui, H. Tao, X. Xu, S. Yang, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", *Journal of Manufacturing Systems*, 78 (2025) 124-152.
3. P. Wang, Y. You, W. Liu, D. Ding, F. Xu, "Knowledge-guided deep reinforcement learning framework for feature machining step sequencing", *Int. J. Adv. Manuf. Technol.*, 140 (2025) 5617-5636.
