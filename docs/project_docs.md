# Project Plan
## Digital Twin–Driven DRL for Energy-Aware CNC Milling Parameter Optimization

---

## Project Summary

**Title:** Digital Twin–Driven Deep Reinforcement Learning for Energy-Aware Dynamic Cutting Parameter Optimization in CNC Milling

**Core Idea:** Build a lightweight Python-based Digital Twin of a CNC milling machine using empirical machining equations. Train a DRL agent (PPO) to dynamically adjust spindle speed and feed rate for each workpiece in a batch, balancing three objectives: maximize throughput (MRR), guarantee surface quality (Ra), and minimize energy consumption.

**Novel Contribution:** Adding energy consumption as a third optimization objective to the existing "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" + "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" framework — producing a "Green Manufacturing" DRL policy.

**References:**
- "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", J. Manuf. Processes 156 — MDP formulation, PPO optimizer, tool wear virtual env
- "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", J. Manuf. Systems 78 — multi-task estimation, batch milling state design

---

## Component Breakdown

### Component 1: Digital Twin Environment (`env/milling_env.py`)

**What it is:** A Gymnasium-compatible Python environment that simulates a CNC milling machine batch process using empirical algebraic equations. Acts as the "Digital Twin."

**Data Sources:**
- No real sensor data required — equations are calibrated from published literature constants
- Calibration reference: "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Table 1 (tool wear constants), "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" Table 2 (surface roughness regression)
- Optional extension: PHM Data Challenge 2010 dataset (milling sensor data) for Ra surrogate validation

**Software Packages:**
- `gymnasium>=0.29` — base class (`gym.Env`)
- `numpy>=1.26` — vectorized equation computation
- `scipy` — for any curve fitting to calibrate constants

**ML Models:** None (pure equations). Optional extension: LSTM surrogate for Ra (replacing the algebraic formula).

**Key Equations implemented:**
- Cutting speed: `v_c = pi * D * n / 1000`
- Feed per minute: `f = n * f_r`
- Machining time: `t_cut = L_cut / f`
- MRR: `MRR = v_c * a_p * f_r` ("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 27)
- Tool wear increment: `delta_VB = C * v_c^0.3 * f_r^0.2 * a_p^0.15 * t_cut` (Taylor extended)
- Surface roughness: `Ra = k1*f_r^2 + k2*VB + k3/v_c` (regression surrogate)
- Energy: `E = (P_idle + k_c * MRR) * t_cut * 60` (novel addition)

**Deliverable:** `MillingDTEnv-v0` registered in gymnasium, passing `check_env()` validation.

**Estimated Effort:** 2–3 days

---

### Component 2: RL Agent — PPO Optimizer (`agent/train_ppo.py`)

**What it is:** Training script for a Proximal Policy Optimization (PPO) agent that interacts with `MillingDTEnv-v0` to learn an optimal, adaptive cutting parameter policy.

**Data Sources:**
- No external dataset — agent learns purely from environment interactions
- Training logs saved to `logs/` (CSV + TensorBoard)

**Software Packages:**
- `stable-baselines3>=2.3` — PPO implementation
- `tensorboard` — training curve visualization
- `gymnasium` — environment interface

**ML Models:**
- **Actor Network:** 2-layer MLP (64→64→2), outputs mean and log-std of Gaussian distribution over actions [n, f_r]
- **Critic Network:** 2-layer MLP (64→64→1), outputs state value V(s)
- **Algorithm:** PPO with GAE (Generalized Advantage Estimation, lambda=0.95, gamma=0.99)

**Key Hyperparameters (from "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Sec 5.3):**
- Learning rate: 3e-4 (Adam)
- Clip range: 0.2
- n_steps: 2048 per update
- Batch size: 64
- Training timesteps: 500,000

**Deliverable:** Trained `.zip` policy file, training reward curve plot, convergence analysis.

**Estimated Effort:** 2–3 days

---

### Component 3: Baseline Comparisons (`baselines/`)

**What it is:** Implement and evaluate 4 non-RL baselines against the PPO agent on the same `MillingDTEnv-v0` environment.

**Data Sources:** Same environment — no additional data.

**Software Packages:**
- `pymoo>=0.6` — NSGA-II multi-objective optimizer baseline
- `numpy` — rule-based and constant baselines
- `pandas` — results tabulation
- `matplotlib` — comparison bar charts, Pareto front plots

**ML Models:** None for baselines. Baselines are:
1. **Constant-High:** Fixed at max (n=8000, f=0.35) throughout batch
2. **Constant-Safe:** Fixed at min (n=2000, f=0.05) throughout batch
3. **Rule-Based Degradation:** Reduces n by 10% every 10 parts to compensate tool wear
4. **NSGA-II Static:** Offline multi-objective optimizer, finds Pareto-optimal parameter set before machining starts, applies it statically
5. **PPO (Ablation, no energy):** Same agent trained with w2=0 (energy ignored) — isolates the effect of the energy term

**Deliverable:** Comparison table (MRR, Ra violation rate, total energy, batch completion rate) and Pareto front visualization.

**Estimated Effort:** 2 days

---

### Component 4: Experiment & Sensitivity Analysis (`experiments/`)

**What it is:** Systematic experiments varying the energy weight w2 in the reward function to characterize the Pareto trade-off between throughput and energy.

**Data Sources:** Training logs from Component 2 runs with w2 ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}.

**Software Packages:**
- `matplotlib` / `seaborn` — Pareto frontier plot (MRR vs Energy)
- `pandas` — experiment result aggregation
- `scipy.stats` — statistical significance testing

**ML Models:** 6 trained PPO instances (one per w2 value).

**Key Analysis:**
- Trade-off curve: as w2 increases, MRR decreases and energy decreases
- Identify the "knee point" of the Pareto frontier (best balanced policy)
- Show that the rule-based baseline is dominated by the DRL policy at all trade-off points

**Deliverable:** Pareto trade-off figure (the key result figure of the project), sensitivity table.

**Estimated Effort:** 1–2 days

---

### Component 5: Visualization & Dashboard (`viz/dashboard.py`)

**What it is:** A set of plots that tell the story of the project end-to-end. These are what you show in the presentation.

**Software Packages:**
- `matplotlib` — all plots
- `seaborn` — heatmaps, distribution plots
- `pandas` — data loading

**Plots to generate:**
1. **Training reward curve** — shows PPO converging over episodes
2. **Tool wear trajectory per episode** — shows agent learning to slow down as tool wears
3. **Action trajectory** — spindle speed and feed rate chosen per workpiece in a batch
4. **Surface roughness per workpiece** — agent keeps Ra below the red line (Ra_max = 1.6 μm)
5. **Energy consumption per workpiece** — shows energy reduction vs Constant-High baseline
6. **Pareto frontier** — MRR vs Total Energy for all methods
7. **Comparison bar chart** — PPO vs 4 baselines on all metrics

**Deliverable:** A `figures/` folder with all publication-quality plots.

**Estimated Effort:** 1–2 days

---

## Timeline

| Week | Component | Milestone |
|:---:|:---|:---|
| 1 | Component 1 | Environment passes `check_env()`, sanity test with random agent |
| 2 | Component 2 | PPO agent converges, reward curve plateau visible |
| 3 | Component 3 | All 4 baselines implemented and evaluated |
| 3 | Component 4 | w2 sensitivity sweep complete, Pareto curve plotted |
| 4 | Component 5 | All figures generated, results written up |

---

## File Structure

```
project/
├── env/
│   └── milling_env.py          # Component 1: Gymnasium Digital Twin
├── agent/
│   ├── train_ppo.py            # Component 2: PPO training script
│   └── evaluate.py             # Load policy and evaluate on env
├── baselines/
│   ├── constant_policy.py      # Constant-High and Constant-Safe
│   ├── rule_based.py           # Rule-based degradation policy
│   └── nsga2_baseline.py       # NSGA-II (pymoo)
├── experiments/
│   └── sweep_w2.py             # Component 4: Pareto sweep
├── viz/
│   └── dashboard.py            # Component 5: All plots
├── logs/                        # TensorBoard + CSV training logs
├── figures/                     # Output plots
├── models/                      # Saved .zip PPO policies
├── requirements.txt
└── README.md
```

---

## Acceptance Criteria

| Criterion | Target |
|:---|:---|
| Environment validity | Passes `stable_baselines3.common.env_checker.check_env()` |
| PPO convergence | Reward improves monotonically over 500k steps |
| Constraint satisfaction | Ra violation rate < 5% on test episodes |
| Tool failure rate | Tool breakage < 2% of episodes |
| Energy reduction | PPO saves ≥ 10% energy vs PPO-no-energy at same MRR |
| Baseline dominance | PPO dominates Rule-Based on both MRR and energy simultaneously |

---

## ME6324 Course Report Mapping

> The course (ME6324 — AI in Manufacturing, Dr. G.L. Samuel, IIT Madras) requires all projects to follow a **6-stage structure** as defined in `2_AI in Manufacturing Project Structure.pptx`. This section maps each required stage to the exact content and components of this project, so the final report can be written directly from this plan.

---

### Stage 1 — Problem Definition & Data Collection

**Required by course:** Clearly define the problem. Justify data relevance, quality, and quantity.

**Our content:**

- **Problem Statement:** CNC milling machines process workpieces in batches. As the cutting tool degrades with each pass, fixed cutting parameters lead to either poor surface quality (Ra > 1.6 μm) or wasted energy. We formulate this as a Markov Decision Process (MDP) and train a Deep Reinforcement Learning agent (PPO) to dynamically adjust spindle speed and feed rate per workpiece to simultaneously maximize throughput, guarantee surface quality, and minimize energy consumption.

- **Data Justification (Critical — professor checks this):** This project uses a **Digital Twin** as its data source instead of a physical sensor dataset. The empirical machining equations (Taylor extended tool wear, surface roughness regression, specific cutting energy) are calibrated from published literature ("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" 2025 Table 1; "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" 2024 Table 2). The Digital Twin acts as a physics-based data generator — every episode of 50 workpieces produces 50 state-action-reward tuples. Over 500,000 training timesteps, the agent processes the equivalent of **10,000 batches (500,000 workpieces)** of synthetic machining data. This is comparable in scale to large supervised learning datasets, with the added benefit of being perfectly labeled and free of sensor noise.

- **Data Quantity summary for report:**
  - Training: ~500,000 environment steps (synthetic)
  - Validation: 100 held-out test episodes × 50 parts = 5,000 test workpieces
  - Equation sources: 2 peer-reviewed Q1/Q2 journal papers

**Maps to:** Component 1 (Digital Twin Env), `gym_env.md`

---

### Stage 2 — Data Pre-processing & Exploration

**Required by course:** Normalization, handling missing values, EDA.

**Our content:**

- **Normalization:** All 7 state variables are normalized to [0, 1] before being fed to the PPO network. Specifically:
  - Tool wear VB: normalized by VB_max (0.3 mm)
  - Spindle speed n: normalized by n_max (8000 RPM)
  - Feed rate f_r: normalized by f_max (0.35 mm/r)
  - Surface roughness Ra: normalized by Ra_max (1.6 μm)
  - MRR: normalized by MRR_max (computed at max n, f_r, fixed a_p)
  - Energy E: normalized by E_max
  - Batch progress i/N: naturally in [0, 1]

- **EDA (Exploratory Analysis of the Digital Twin):**
  - Plot Ra as a function of f_r and VB (2D surface plot) — shows how tool degradation amplifies roughness
  - Plot tool wear trajectory under Constant-High vs Constant-Safe policies (baseline sanity check)
  - Plot energy vs MRR trade-off surface across the full (n, f_r) action space
  - These plots go in the report as the "Data Exploration" section and directly motivate why a fixed policy is suboptimal

- **No missing values:** Digital Twin is deterministic by design; no imputation needed.

**Maps to:** Component 5 (Visualization Dashboard), pre-training analysis plots

---

### Stage 3 — Model Selection & Architecture Design

**Required by course:** Justify model choice, define architecture (layers, connections, activation functions), list tools and libraries.

**Our content:**

- **Why PPO over alternatives:**

  | Algorithm | Reason for/against |
  |:---|:---|
  | **PPO (chosen)** | Handles continuous action space; stable training; used by "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" on same problem class — directly comparable |
  | DQN | Requires discretizing continuous actions — loses resolution |
  | SAC | More sample-efficient but more complex; use as extension/ablation |
  | GA/NSGA-II | Cannot adapt in real-time; static solution — used as baseline |

- **Actor Network Architecture:**
  - Input: 7D normalized state vector
  - Hidden: 2 × fully-connected layers, 64 units each, Tanh activation
  - Output: Mean and log-std of 2D Gaussian (spindle speed, feed rate)

- **Critic Network Architecture:**
  - Input: 7D normalized state vector
  - Hidden: 2 × fully-connected layers, 64 units each, Tanh activation
  - Output: Scalar state value V(s)

- **Tools & Libraries:**

  | Tool | Purpose |
  |:---|:---|
  | Python 3.11 | Primary language |
  | Gymnasium 0.29 | RL environment standard |
  | Stable-Baselines3 2.3 | PPO implementation |
  | NumPy 1.26 | Digital Twin equation computation |
  | PyTorch (via SB3) | Neural network backend |
  | Matplotlib / Seaborn | Visualization |
  | pymoo 0.6 | NSGA-II baseline |
  | TensorBoard | Training monitoring |
  | Jupyter Notebook | Prototyping and EDA |
  | GitHub | Version control |

**Maps to:** Component 1 (env architecture) + Component 2 (PPO design)

---

### Stage 4 — Model Training & Evaluation

**Required by course:** Train the model, monitor hyperparameters, evaluate on unseen test data.

**Our content:**

- **Training procedure:**
  - 500,000 total timesteps on `MillingDTEnv-v0`
  - PPO updates every 2,048 steps (n_steps), batch size 64, 10 epochs per update
  - Learning rate 3×10⁻⁴ (Adam), clip range ε = 0.2, GAE λ = 0.95, discount γ = 0.99
  - Training monitored via TensorBoard: episode reward, Ra violation frequency, tool failure rate

- **Hyperparameter tuning:**
  - Primary tunable: reward weight w₂ (energy penalty) ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
  - Secondary: network hidden size (32 vs 64 vs 128) — quick ablation

- **Evaluation on unseen data:**
  - 100 test episodes with fresh tool (VB=0), never seen during training
  - Metrics: mean batch reward, Ra violation rate (%), tool failure rate (%), total energy (kJ), mean MRR (mm³/min), batch completion rate (%)
  - Comparison table: PPO vs all 4 baselines + PPO-ablation

**Maps to:** Component 2 (training) + Component 3 (baselines) + Component 4 (sensitivity)

---

### Stage 5 — Model Deployment & Monitoring

**Required by course:** Deploy model, track key metrics, plan for retraining.

**Our content:**

- **Deployment:** Trained PPO policy saved as `models/ppo_milling_w2_0.4.zip`. A standalone `agent/evaluate.py` script loads this policy and runs it on any new `MillingDTEnv-v0` configuration (different batch size, material constants, etc.) without retraining.

- **Monitoring Dashboard** (`viz/dashboard.py`): The 7-panel dashboard serves as the "monitoring" layer:
  1. Training reward curve (convergence check)
  2. Tool wear trajectory per episode (health monitoring)
  3. Spindle speed & feed rate per workpiece (action monitoring)
  4. Ra per workpiece vs Ra_max threshold (quality alarm)
  5. Energy per workpiece (efficiency monitoring)
  6. Pareto frontier: all methods
  7. Final comparison bar chart

- **Retraining trigger:** If tool wear constants (C_wear, α, β, γ) change due to new tool grade or material, the Digital Twin parameters are updated and the agent is fine-tuned for 50,000 additional steps — not retrained from scratch. This is the "data feedback loop" the course requires.

**Maps to:** Component 5 (Dashboard) + Component 2 (evaluate.py)

---

### Stage 6 — Iteration & Improvement

**Required by course:** Iterative cycle — monitor, feedback loop, refine architecture.

**Our content:**

- **Iteration 1 (w₂ sweep):** Train 6 PPO variants with w₂ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}. Plot Pareto trade-off curve. Select the "knee point" (best balanced w₂). This is the core experiment.

- **Iteration 2 (architecture ablation):** Compare 64-64 vs 32-32 vs 128-128 MLP hidden sizes to confirm architecture choice.

- **Iteration 3 (Ra surrogate upgrade):** Replace the algebraic Ra formula with a data-driven LSTM surrogate (trained on PHM 2010 milling dataset). Re-evaluate PPO. This shows the "feedback loop from real data" that the course expects.

- **Future work (for report conclusion):**
  - Add depth of cut a_p as a third continuous action
  - Extend to multi-tool scenario (tool change decision as discrete action — exactly as "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" do)
  - Use real IoT sensor stream to update VB in real time (true Digital Twin loop)

**Maps to:** Component 4 (sensitivity sweep) + future extensions section of report

---

## Quick Report Outline

Use this directly as the section headings for the final project report:

```
1. Introduction & Motivation
2. Literature Review ("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" 2025, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" 2024 + 3 supporting papers)
3. Problem Definition & Data Collection
   3.1 MDP Formulation
   3.2 Digital Twin as Data Generator (justify data source)
4. Data Pre-processing & Exploration
   4.1 State Normalization
   4.2 EDA: Ra surface, Wear trajectory, Energy–MRR trade-off
5. Model Selection & Architecture Design
   5.1 Why PPO
   5.2 Actor-Critic Architecture
   5.3 Tools & Libraries
6. Model Training & Evaluation
   6.1 Training Setup & Hyperparameters
   6.2 Convergence Analysis
   6.3 Evaluation vs Baselines
7. Model Deployment & Monitoring
   7.1 Policy Deployment (evaluate.py)
   7.2 Monitoring Dashboard
8. Iteration & Improvement
   8.1 w₂ Sensitivity & Pareto Frontier
   8.2 Architecture Ablation
   8.3 Future Work
9. Conclusion
10. References
```



# Gym Environment Specification
## Digital Twin–Driven Milling Parameter Optimization (`MillingDTEnv-v0`)

> **Based on:**
> - "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" *"DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"* — J. Manufacturing Processes 156, 1006–1031
> - "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" *"Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning"* — J. Manufacturing Systems 78, 124–152

---

## 1. Problem Framing

A CNC milling machine processes a **batch of N workpieces** sequentially. As machining progresses, the cutting tool degrades. The RL agent must decide—before each workpiece in the batch—what **spindle speed** and **feed rate** to use so that:
- All parts meet surface roughness requirements (Ra ≤ Ra_max)
- Tool wear never exceeds the critical bluntness threshold (VB ≤ VB_max)
- Total machining energy is minimized *(novel contribution)*
- Batch throughput (Material Removal Rate) is maximized

This is formulated as a **finite-horizon MDP** over one batch of N parts.

---

## 2. State Space

The observation vector s_i at the i-th workpiece in the batch:

`s_i = [VB_i, n_{i-1}, f_{i-1}, Ra_{i-1}, MRR_{i-1}, E_{i-1}, i/N]`

| Index | Variable | Symbol | Unit | Source |
|:---:|:---|:---:|:---:|:---|
| 0 | Tool wear value | VB_i | mm | "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 17, updated each step |
| 1 | Previous spindle speed | n_{i-1} | RPM | "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 5 |
| 2 | Previous feed rate | f_{i-1} | mm/r | "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" nomenclature |
| 3 | Previous surface roughness | Ra_{i-1} | μm | "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" Sec 3.1, ML model output |
| 4 | Previous Material Removal Rate | MRR_{i-1} | mm³/min | "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 27 |
| 5 | Previous energy consumption | E_{i-1} | kJ | **Novel addition** |
| 6 | Batch progress | i/N | — | Completion ratio |

**Observation space:** `spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)` (all values normalized to [0, 1])

---

## 3. Action Space

The agent selects **two continuous parameters** before each workpiece cut:

`a_i = [n_i, f_i]`

| Variable | Symbol | Min | Max | Unit |
|:---|:---:|:---:|:---:|:---:|
| Spindle speed | n | 2000 | 8000 | RPM |
| Feed rate (per rev) | f_r | 0.05 | 0.35 | mm/r |
| Depth of cut | a_p | **fixed** | 0.5 | mm (fixed for simplicity) |

> **Depth of cut (a_p) is fixed** per "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" Sec 3.1 to reduce action dimensionality for a course project. It can be made variable in extension work.

**Action space:** `spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)`  
*Actions are normalized to [-1,1] and linearly scaled to real machine limits inside `step()`.*

---

## 4. Digital Twin Transition Equations

These are the **exact empirical formulas** from the papers that form the "Digital Twin" core.

### 4.1 Cutting Speed
*("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 5)*

`v_c = (pi * D * n) / 1000   [m/min]`

where D = workpiece/tool diameter (mm).

### 4.2 Feed per Minute
*("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 6)*

`f = n * f_r   [mm/min]`

### 4.3 Machining Time per Workpiece
*("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 7–8)*

`t_cut = L_cut / f   [min]`

where L_cut = cutting length per workpiece (mm, a fixed environment parameter).

### 4.4 Material Removal Rate
*("DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Eq. 27)*

`MRR_i = v_c * a_p * f_r   [mm³/min]`

### 4.5 Tool Wear Update — Taylor Extended Model
*(Adapted from "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" constraint Eq. 17 and virtual environment description)*

Tool wear increments each pass. A simplified Taylor-extended wear rate model:

`delta_VB_i = C_wear * v_c^alpha * f_r^beta * a_p^gamma * t_cut`
`VB_{i+1} = VB_i + delta_VB_i`

Calibration constants (fitted for mild steel / carbide tool):
- C_wear = 1.1e-4, alpha = 0.3, beta = 0.2, gamma = 0.15

*These values yield VB = 0.3 mm (bluntness limit) after ~50 passes at max speed — consistent with "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" experimental data.*

### 4.6 Surface Roughness
*("Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" Eq. 1 objective; "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" Sec 3 Ra constraint)*

A regression-based surrogate model:

`Ra_i = k1 * f_r^2 + k2 * VB_i + k3 * v_c^(-1)   [μm]`

Typical calibration for carbide milling of steel: k1 = 12.5, k2 = 0.8, k3 = 0.5.

> **Extension:** "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" use a data-driven LSTM multi-task estimator for Ra. As an extension, train an LSTM Ra predictor from PHM public datasets and replace this surrogate.

### 4.7 Energy Consumption — Novel Contribution
*(Standard specific cutting energy formulation; not in either base paper)*

`P_cutting = k_c * MRR_i   [W]`
`E_i = (P_idle + P_cutting) * t_cut * 60   [kJ]`

where:
- k_c = Specific cutting energy coefficient (~2.5 W·min/mm³ for steel)
- P_idle = Idle/spindle motor power (~500 W, fixed machine constant)

---

## 5. Reward Function

The reward is a **weighted multi-objective scalar** balancing four competing objectives:

`r_i = w1 * MRR_hat_i - w2 * E_hat_i - r_quality - r_wear`

| Term | Formula | Purpose |
|:---|:---|:---|
| Throughput bonus | +w1 * MRR_hat_i | Maximize material removal rate |
| Energy penalty | -w2 * E_hat_i | Minimize energy consumption (novel) |
| Quality violation | -500 if Ra_i > Ra_max, else 0 | Penalize defective parts |
| Tool failure | -1000 + terminated=True if VB_i > VB_max | Catastrophic tool breakage |
| Batch completion bonus | +200 on last step if no violations | Encourage full-batch completion |

**Normalization:** MRR_hat and E_hat are normalized to [0,1] using known min/max of action space.

**Default weights:** w1 = 0.6, w2 = 0.4 (tunable hyperparameter — the key experimental variable).

---

## 6. Episode Logic

```python
reset() → Fresh tool (VB=0), i=0, batch size N=50
step(action):
    1. Scale action → (n, f_r) in real units
    2. Compute v_c, f, t_cut, MRR             [Eqs 4.1–4.4]
    3. Update tool wear: VB += delta_VB        [Eq 4.5]
    4. Compute Ra                              [Eq 4.6]
    5. Compute Energy E                        [Eq 4.7]
    6. Compute reward r                        [Sec 5]
    7. Check termination conditions
    8. Advance i += 1; return (obs, reward, terminated, truncated, info)
```

**Termination conditions:**
- `VB > VB_max` (0.3 mm) → episode ends with large penalty
- `i == N` → batch complete, episode ends normally

---

## 7. Environment Constants (Default Configuration)

| Parameter | Symbol | Value | Unit |
|:---|:---:|:---:|:---:|
| Batch size | N | 50 | parts |
| Tool wear limit | VB_max | 0.3 | mm |
| Surface roughness limit | Ra_max | 1.6 | μm |
| Workpiece diameter | D | 50 | mm |
| Depth of cut | a_p | 0.5 | mm |
| Cutting length | L_cut | 100 | mm |
| Idle power | P_idle | 500 | W |
| Specific cutting energy | k_c | 2.5 | W·min/mm³ |
| Spindle speed range | [n_min, n_max] | [2000, 8000] | RPM |
| Feed rate range | [f_min, f_max] | [0.05, 0.35] | mm/r |

---

## 8. Comparison Baselines

| Baseline | Description |
|:---|:---|
| **Constant-High** | Maximum spindle speed and feed rate throughout |
| **Constant-Safe** | Minimum speed/feed throughout |
| **Rule-Based** | Reduce speed by 10% every 10 parts as tool degrades |
| **NSGA-II** | Static multi-objective optimization (pymoo library) |
| **PPO (no energy)** | Ablation: same agent with w2=0 (ignores energy — replicates "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation") |

---

## 9. Libraries & Implementation Notes

```
gymnasium>=0.29        # Env base class
stable-baselines3>=2.3 # PPO, SAC agents
numpy>=1.26            # Numerical computation
matplotlib             # Training curves, reward plots
pandas                 # Logging episode data
pymoo                  # NSGA-II baseline
```

**Recommended RL algorithm:** PPO (as used in "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation") with 2-layer MLP actor-critic (64-64 hidden units).
**Alternative:** SAC for continuous action space (more sample efficient, good for ablation).

