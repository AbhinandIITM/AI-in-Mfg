# Digital Twin-Driven DRL for Energy-Aware CNC Milling Parameter Optimization

**Course:** ME6324 - AI in Manufacturing (IIT Madras)

## Overview
This project applies Deep Reinforcement Learning (DRL) to address the dynamic cutting parameter optimization problem in CNC face milling operations. Utilizing a high-fidelity Python-based Digital Twin grounded in empirical manufacturing physics, an autonomous agent dynamically optimizes cutting parameters across multi-pass machining:
- **Spindle Speed ($n$):** 800 - 3200 RPM
- **Feed per Tooth ($f_r$):** 0.05 - 0.25 mm/tooth
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
  - Two hidden layers of 64 units each with Tanh activations for both policy ($\pi_\theta$) and value function ($V_\phi$).
- **Reward Formulation:**
  $$r_t = w_1 \frac{MRR}{MRR_{\text{max}}} - w_2 \frac{E}{E_{\text{max}}} + r_{\text{progress}} + r_{VB} + r_B + r_{Ra} + r_{\text{step}} + r_{\text{HV}}$$
  - Throughput vs. Energy Trade-off: Controlled by weights $w_1$ and $w_2$.
  - Progress Bonus: $r_{\text{progress}} = 10 \times \Delta \text{Stock Removed Ratio}$.
  - Constraint Penalties:
    - $r_{VB} = -50.0$ if $VB > VB_{\text{safe}}$
    - $r_B = -100.0$ if catastrophic tool breakage occurs
    - $r_{Ra} = -50.0$ if final pass $Ra > Ra_{\text{safe}}$ ($+50.0$ if within safe spec)
    - Step penalty: $r_{\text{step}} = -0.05$ per pass
  - Hypervolume & Success Bonus: $+200.0$ for successful stock completion within specs, plus Pareto-efficiency hypervolume bonus based on cycle time and final roughness.

---

## Repository Structure

```text
.
|-- agent/
|   |-- train_rl.py              # Primary PPO training pipeline
|   |-- train_rl_ablation.py     # Energy weight (w2 in {0.0, 0.5, 1.0}) ablation study
|   |-- evaluate.py              # Monte Carlo statistical robustness evaluation
|   |-- run_pipeline.py          # Master execution script (train, eval, ablate, plot)
|   |-- run_experiments.py       # Parallel worker script for multi-seed runs
|-- baselines/
|   |-- genetic_algorithm.py     # Multi-objective NSGA-II baseline
|   |-- rule_based.py            # Industry heuristic and conservative static baseline
|-- config/
|   |-- physics_constants.py     # Centralized physical, machine, and material constants
|-- env/
|   |-- milling_env.py           # Gymnasium-compliant CNC Milling Digital Twin
|-- results/
|   |-- ablation/                # Ablation evaluation metrics and CSV summaries
|   |-- monte_carlo/             # Monte Carlo sweep trajectory datasets
|   |-- plots/                   # Generated evaluation plots and figures
|   |-- tensorboard/             # TensorBoard training curves and scalars
|-- models/
|   |-- PPO_seed42/              # Primary trained model policy and normalization stats
|   |-- ablation_w2_*/           # Models trained across discrete energy weight configurations
|-- viz/
|   |-- plot_eval.py             # Generates KDE distributions and failure landscape plots
|   |-- plot_results.py          # Parses TensorBoard logs for reward curves
|   |-- plot_baselines_comparison.py # Comparative analysis between PPO and classical baselines
|-- app.py                       # Streamlit interactive CNC Digital Twin workstation
`-- tests/                       # Pytest test suite for physical consistency
```

---

## Execution Guide

### 1. Master Pipeline (Recommended)
To run the full workflow—primary PPO training (1,000,000 steps), Monte Carlo validation (100 episodes), discrete weight ablation study, and publication-ready plot generation:
```bash
python agent/run_pipeline.py --steps 1000000 --ablation_steps 1000000
```

### 2. Standalone Training & Evaluation
- **Train Primary Policy:**
  ```bash
  python agent/train_rl.py --algo PPO --steps 1000000 --seed 42
  ```
- **Monte Carlo Statistical Sweep:**
  ```bash
  python agent/evaluate.py
  ```
- **Energy Weight Ablation Study:**
  ```bash
  python agent/train_rl_ablation.py --steps 1000000
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

---

## Experimental Findings

1. **Statistical Robustness:** In a 100-episode Monte Carlo evaluation with stochastic initial tool wear and tool breakage hazards, the trained PPO policy achieved a 100.0% completion rate with 0.0% violations of surface roughness ($Ra$) and flank wear ($VB$).
2. **Dynamic Multi-Pass Strategy:** The agent utilizes aggressive cutting parameters ($a_p \approx 2.0 - 2.5\ \text{mm}$, high feed rate) during roughing passes to maximize MRR, transitioning to high spindle speeds and reduced feeds during final passes to ensure surface finish compliance.
3. **Ablation Dynamics:** Under strict safety penalties, the policy prioritizes boundary compliance, converging to stable, robust cutting regimes across discrete energy weighting variations ($w_2 \in \{0.0, 0.5, 1.0\}$).

---

## References

1. **DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation**, *Journal of Manufacturing Processes*, Vol. 156.
2. **Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning**, *Journal of Manufacturing Systems*, Vol. 78.
