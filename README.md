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
  $$P_{\text{total}} = P_{\text{idle}} + k_c \cdot MRR$$
  $$E_{\text{cut}} = P_{\text{total}} \cdot t_{\text{cut}}$$

### Calibrated Constants
All values live in `config/physics_constants.py` and are shared by the environment, the baselines and the ablation study, so no consumer can drift from another.

| Group | Coefficients | Values |
| :--- | :--- | :--- |
| Tool wear | $C_{\text{wear}}, \alpha, \beta, \gamma_{\text{wear}}$ | 1.1e-4, 0.3, 0.2, 0.15 |
| Surface roughness | $k_1, k_2, k_3$ | 12.5, 0.8, 0.5 |
| Energy | $P_{\text{idle}}, k_c$ | 500 W, 2.5 W$\cdot$min/mm$^3$ |
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
- **State Representation (8-dim):**
  $$s_t = \left[ \widetilde{VB}, \widetilde{\kappa}, \tilde{n}_{\text{prev}}, \tilde{f}_{\text{prev}}, \tilde{a}_{p,\text{prev}}, \widetilde{MRR}_{\text{prev}}, \widetilde{Ra}_{\text{prev}}, \mathbb{I}_{\text{broken}} \right]$$
  Observations utilize soft tanh-squashing for continuous normalized gradients.
- **Action Space (3-dim continuous):**
  $$a_t = [a_n, a_f, a_p] \in [-1, 1]^3 \mapsto [n, f_r, a_p]$$
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

### Conditioned Policy
The observation of `MillingDTEnv-w2-v1` carries the energy weight and the
material context next to the eight physics features:
$$s_t = [\widetilde{VB}, \widetilde{\kappa}, \tilde{n}_{\text{prev}}, \tilde{f}_{\text{prev}}, \tilde{a}_{p,\text{prev}}, \widetilde{MRR}_{\text{prev}}, \widetilde{Ra}_{\text{prev}}, \mathbb{I}_{\text{broken}}, w_2, \widetilde{C_{\text{wear}}}, \widetilde{k_1}]$$
Both the weight and the scenario are drawn per episode during training, so a single policy covers the whole weight range and the whole material distribution. At runtime the weight is a query, not a retrain. `agent/evaluate_conditioned.py` sweeps it and compares the resulting frontier against the per-weight sweep.

### Domain Randomization
`MillingDTEnv(randomize=True)` draws a new machining scenario for every part: the stock allowance, the tool diameter, the cutting length, the wear and roughness coefficients, and the initial flank wear of the tool. The bands live in `config/physics_constants.py` and are shared with the Monte Carlo sweep, so the policy is trained on the distribution it is measured on.

### Receding-Horizon Controller
`baselines/receding_horizon.py` re-solves the next pass at every step, with the remaining stock and the current flank wear as the state, and keeps the candidate that scores best and stays inside the roughness and wear limits. It is a stronger classical baseline than the static NSGA-II solution: the four static baselines all end outside the roughness safety limit, and this one is the only classical policy that stays inside it.

### Transfer to a New Material
`agent/transfer_experiment.py` measures how much of a new material a policy that was trained on another one already knows. The same target task (a harder alloy, a faster wearing tool) is evaluated three ways: the source policy zero shot, the source policy fine tuned, and a policy trained on the target from scratch for the same budget.

---

## Repository Structure

```text
.
|-- agent/
|   |-- evaluate.py              # Monte Carlo robustness sweep with 95% confidence intervals
|   |-- evaluate_conditioned.py  # Weight sweep of the conditioned policy against the per-weight sweep
|   |-- run_experiments.py       # Parallel supervisor for multi-seed runs
|   |-- run_pipeline.py          # Master execution script (train, eval, ablate, plot)
|   |-- summarize_results.py     # Builds the result tables quoted in the docs from the artifacts
|   |-- transfer_experiment.py   # Zero shot, fine tuned and from scratch on a new material
|   |-- train_rl.py              # Primary PPO training pipeline
|   |-- train_rl_ablation.py     # Energy weight ablation (7 weights x 3 seeds)
|   |-- train_rl_conditioned.py  # Conditioned policy over the weight and the scenario
|-- app.py                       # Streamlit interactive CNC Digital Twin workstation
|-- baselines/
|   |-- receding_horizon.py      # Controller that re-solves at every pass
|   |-- run_baselines.py         # Constant-high, constant-safe, rule-based, NSGA-II and receding horizon
|-- config/
|   |-- physics_constants.py     # Centralized physical, machine, and material constants
|   |-- scenarios.py             # Randomized machining scenarios, shared by training and evaluation
|-- docs/
|   |-- ENVIRONMENT_SPEC.md      # The environment contract and the acceptance status
|   |-- INTERACTIVE_SIMULATOR_PLAN.md # Design of the Streamlit workstation
|   |-- LITERATURE_COMPARISON.md # Position against the three papers in docs/papers/
|   |-- METHOD_UPGRADE_PLAN.md   # Conditioned policies, transfer, and the controller teacher
|   |-- references.md             # Where every physics equation comes from
|-- env/
|   |-- __init__.py              # Single registration point for the environments
|   |-- milling_env.py           # Gymnasium-compliant CNC Milling Digital Twin
|-- models/
|   |-- PPO_seed42/              # Primary trained model policy and normalization stats
|   |-- PPO_conditioned_seed42/  # Policy conditioned on the energy weight and the material
|   |-- ablation_w2_*_seed*/     # Models trained across the energy weight grid
|-- results/
|   |-- ablation/                # Ablation evaluation metrics and CSV summaries
|   |-- baseline_results_v5.csv  # Classical baseline metrics
|   |-- conditioned/             # Frontier of the conditioned policy
|   |-- curves/                  # Training curves, one progress.csv per run
|   |-- monte_carlo/             # Monte Carlo sweep per-scenario datasets
|   |-- plots/                   # Generated evaluation plots and figures
|   |-- transfer/                # Transfer experiment results
|   |-- RESULTS.md               # Generated summary of every artifact
|-- sim/
|   |-- machining_sim.py         # Trajectory simulation and G-code synthesis for the app
|-- tests/                       # Pytest test suite for physical consistency
|-- utils/
|   |-- seeding.py               # Centralized reproducible seeding
`-- viz/
    |-- plot_baselines_comparison.py # Comparative analysis between PPO and classical baselines
    |-- plot_eval.py             # Generates KDE distributions and failure landscape plots
    |-- plot_results.py          # Plots the training curves from the progress.csv files
    |-- plotting_utils.py        # Shared plotting helpers, single source for figure style
```

---

## Execution Guide

### 1. Master Pipeline (Recommended)
To run the full workflow (primary PPO training, Monte Carlo validation with confidence intervals, the energy weight ablation, the classical baselines, and plot generation):
```bash
python agent/run_pipeline.py --steps 1000000 --ablation_steps 100000
```

### 2. Standalone Training & Evaluation
- **Train Primary Policy** (on randomized machining scenarios):
  ```bash
  python agent/train_rl.py --algo PPO --steps 1000000 --seed 42 --randomize
  ```
- **Train the Conditioned Policy** (one policy over the energy weight and the material):
  ```bash
  python agent/train_rl_conditioned.py --steps 1000000 --seed 42
  ```
- **Monte Carlo Statistical Sweep:**
  ```bash
  python agent/evaluate.py
  ```
- **Energy Weight Ablation Study** (7 energy weights, 3 seeds each, trained in parallel):
  ```bash
  python agent/train_rl_ablation.py --steps 150000 --scenarios 50 --slots 7
  ```
- **Classical Baselines** (including the receding-horizon controller):
  ```bash
  python baselines/run_baselines.py
  ```
- **Transfer to a new material** (zero shot, fine tuned, from scratch):
  ```bash
  python agent/transfer_experiment.py 300000 50000
  ```
- **Rebuild the result tables** quoted in this file:
  ```bash
  python agent/summarize_results.py
  ```

### 3. Generate Visualizations
- **Training Convergence Curves:**
  ```bash
  python viz/plot_results.py
  ```
- **KDE Constraint Distributions & Failure Landscapes:**
  ```bash
  python viz/plot_eval.py
  ```
- **Comparative Baseline Benchmarks:**
  ```bash
  python viz/plot_baselines_comparison.py
  ```
All output figures are saved to `results/plots/`.

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
processes (`agent/run_experiments.py`, 7 slots by default), which is what
uses the cores. A GPU would pay off for a convolutional policy, a much
larger network, or an off-policy algorithm with a replay buffer.

---

## Experimental Findings

1. **Robustness over 100 randomized scenarios.** The policy is trained across randomized machining scenarios and evaluated on 100 held-out scenarios of the same kind. It completed 98.0% of the parts, kept the finished surface inside the roughness limit in **100%** of them, never exceeded the flank wear limit, and broke a tool in 2%. Material removal rate 1325.7 mm3/min [95% CI 1204.9-1446.5], energy 55.2 kJ per part, cycle time 16.7 s. The worst pass of an episode exceeds the roughness limit in 40% of the scenarios, which is the roughing pass: the reward constrains the finishing pass, because that is the pass that sets the surface of the part.

2. **The energy weight moves the operating point.** Over 7 weights and 3 seeds, the energy of a part falls by 21% (79.1 to 62.4 kJ) and the cycle time by 53% (62.6 to 29.1 s) as the weight goes from 0.0 to 1.0, with the part completion rate between 0.99 and 1.00 throughout. The material removal rate rises over the same range (835 to 952 mm3/min), and that is not a mistake in the experiment: the energy of a part is $k_c V + P_{\text{idle}} T_{\text{part}}$, so the cutting term is fixed by the volume, and the idle term falls exactly when the cycle time falls, which is also what raises the removal rate. In this digital twin the two objectives point the same way, and the energy weight is not a trade-off against throughput. `docs/LITERATURE_COMPARISON.md` states what would be needed to open one.

3. **One conditioned policy replaces 21 weight-specific policies.** At the same training budget per model, the single conditioned policy beats every weight-specific policy: 12% to 22% more material removal rate and 8% to 19% less energy per part, from one training run instead of 21. The energy weight stays a runtime query rather than a retrain.

4. **The classical baselines were weak until the controller was made dynamic.** The static NSGA-II solution, the two constant policies and the rule-based policy all end outside the roughness safety limit and score around -45. The receding-horizon controller, which re-solves at every pass, scores +204.3 and is the strongest baseline in the repository.

5. **Fine tuning does not fix a material change at this budget.** A policy trained on the nominal material scores 184.5 on a harder alloy and violates the roughness limit in every scenario. Fine tuning it for 50,000 steps moves the score to 184.8 and removes no violation. A policy trained on that alloy from scratch for the same 50,000 steps scores 196.0 with no violations at all. For this shift, training on the target distribution beats pre-training and fine-tuning, which is the argument for the domain randomization above.

6. **The learned policy and the classical controller are level.** On the same 50 episodes, the PPO policy scores 204.9 and the receding-horizon controller 204.3, and the controller is ahead on the material removal rate, the energy and the cycle time. The earlier comparison against the static baselines was hollow, because all four of them end outside the roughness safety limit. This digital twin does not yet show a reinforcement learning advantage, and the honest next experiment is to start the policy from the controller and fine-tune from there.

Every number below is generated from the artifacts in `results/` by
`python agent/summarize_results.py`, which writes the same tables to
`results/RESULTS.md`. The methods behind each result are described in
`docs/METHOD_UPGRADE_PLAN.md`, and the position against the published work is
in `docs/LITERATURE_COMPARISON.md`.

---

## References

1. X. Mo, M. Xia, S. Mei, X. Hu, "Deep reinforcement learning-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", *Journal of Manufacturing Processes*, 156 (2025) 1006-1031.
2. P. Wang, Y. Cui, H. Tao, X. Xu, S. Yang, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", *Journal of Manufacturing Systems*, 78 (2025) 124-152.
3. P. Wang, Y. You, W. Liu, D. Ding, F. Xu, "Knowledge-guided deep reinforcement learning framework for feature machining step sequencing", *Int. J. Adv. Manuf. Technol.*, 140 (2025) 5617-5636.
