# Project Requirements Specification (`req.md`)

## Project Title
**Knowledge-Guided Deep Reinforcement Learning for Feature Machining Step Sequencing in Intelligent CAPP**

---

## 1. Executive Summary & Category Definition

* **Course/Domain:** AI in Manufacturing (IIT Madras)
* **Difficulty Category:** **Medium Category** *(Discrete State-Action MDP, Data-Driven & Combinatorial Optimization, Zero Physics/Continuum Mechanics)*
* **Primary Reference Paper:** 
  > *Wang, P., You, Y., Liu, W., Ding, D., & Xu, F. (2025). Knowledge-guided deep reinforcement learning framework for feature machining step sequencing. The International Journal of Advanced Manufacturing Technology, 140, 5617–5636.* [DOI: 10.1007/s00170-025-16606-4]
* **Secondary Reference Papers (from Q1/Q2 dataset):**
  > *"A novel method based on deep reinforcement learning for machining process route planning" Robotics and Computer-Integrated Manufacturing, 86, 102688.* [Q1, IF: 12.3]  
  > *"Employing deep reinforcement learning for machining process planning: An improved framework" Journal of Manufacturing Systems, 78, 102.10.* [Q1, IF: 14.9]

### 1.1 Category Justification (Why Medium & Physics-Free)
* **No Physics Involved:** Avoids complex continuous thermal PDE solvers, cutting force/chatter dynamics differential equations, fluid dynamics (CFD), and finite element modeling (FEA).
* **Pure Decision Intelligence:** Focuses on combinatorial sequencing, discrete manufacturing knowledge graphs, tool/setup optimization, and Markov Decision Processes (MDPs) with dynamic action masking.
* **Academic Rigor:** Addresses NP-hard Computer-Aided Process Planning (CAPP) optimization, benchmarking against Genetic Algorithms (GA), Simulated Annealing (SA), and Priority Dispatching Rules.

---

## 2. Problem Statement & Manufacturing Scope

### 2.1 The Industrial Challenge
In Computer-Aided Process Planning (CAPP) for CNC machining, a part consists of multiple **Machining Features (MFs)** (e.g., planes, steps, slots, pockets, holes). Each feature requires decomposition into multiple **Feature Machining Steps (FMSs)** (e.g., *Rough Milling $\to$ Semi-Finish Milling $\to$ Finish Milling* or *Center Drilling $\to$ Drilling $\to$ Reaming*).

Determining the execution sequence of all FMSs is NP-hard due to:
1. **Complex Precedence Constraints:** Geometric and process dependencies (e.g., planar reference surfaces must be machined before locating holes; roughing steps must precede finishing steps).
2. **Setup and Tool Optimization:** Frequent tool changes, machine transfers, and workpiece orientation (setup) changes drastically inflate non-productive auxiliary times and manufacturing costs.

### 2.2 Project Objective
Develop and train a **Knowledge-Guided Deep Reinforcement Learning (DRL) agent** with **Dynamic Invalid Action Masking** to autonomously generate optimal, 100% valid FMS machining sequences that minimize total manufacturing cost (tool change cost, setup change cost, machine change cost, and non-cutting tool path distance) while outperforming traditional metaheuristics in both solution quality and computational inference time.

---

## 3. Mathematical Formulation (MDP Framework)

The sequencing process is modeled as an episodic finite-horizon Markov Decision Process: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{M}_{\text{mask}} \rangle$.

### 3.1 State Space $\mathcal{S}$
At step $t$, the state $s_t \in \mathcal{S}$ captures the machining progress and current machine tool state:
$$s_t = \left[ \mathbf{v}_{\text{status}}, \mathbf{v}_{\text{current}}, \mathbf{V}_{\text{features}} \right]$$
Where:
* $\mathbf{v}_{\text{status}} \in \{0, 1\}^N$: Binary completion vector for all $N$ Feature Machining Steps ($1$ if completed, $0$ otherwise).
* $\mathbf{v}_{\text{current}} = [m_t, \theta_t, t_t, x_t, y_t, z_t]$: Current machine ID ($m_t$), clamping/spindle orientation ($\theta_t$), loaded cutting tool ID ($t_t$), and tool position coordinates $(x_t, y_t, z_t)$.
* $\mathbf{V}_{\text{features}} \in \mathbb{R}^{N \times d}$: Static feature matrix describing each FMS (required machine type, required orientation, tool type, cutting depth, tolerance level, feature category).

### 3.2 Action Space $\mathcal{A}$ & Dynamic Action Masking $\mathcal{M}_{\text{mask}}$
* The action $a_t \in \mathcal{A} = \{1, 2, \dots, N\}$ corresponds to selecting the next FMS to execute.
* **Knowledge-Guided Action Mask $\mathbf{m}(s_t) \in \{0, 1\}^N$:**
  $$m_i(s_t) = \begin{cases} 1, & \text{if FMS}_i \text{ is unexecuted AND all predecessors } \text{Pred}(i) \text{ are completed} \\ 0, & \text{otherwise (invalid action)} \end{cases}$$
* **Policy Masking:** Softmax logits for invalid actions ($m_i = 0$) are set to $-\infty$, guaranteeing $100\%$ feasible sequences with zero constraint violations.

### 3.3 State Transition $\mathcal{P}(s_{t+1} \mid s_t, a_t)$
Executing action $a_t = k$:
1. Updates status: $\mathbf{v}_{\text{status}}[k] \leftarrow 1$.
2. Updates tool state: $m_{t+1} \leftarrow m(k)$, $\theta_{t+1} \leftarrow \theta(k)$, $t_{t+1} \leftarrow t(k)$, and position $(x, y, z)_{t+1} \leftarrow \text{pos}(k)$.
3. Computes updated precedence dependencies for remaining steps.

### 3.4 Multi-Objective Cost & Reward Function $\mathcal{R}(s_t, a_t, s_{t+1})$
The cost of executing step $a_t$ from state $s_t$ is:
$$C(s_t, a_t) = w_1 \cdot C_{\text{machine}}(m_t, m_{a_t}) + w_2 \cdot C_{\text{setup}}(\theta_t, \theta_{a_t}) + w_3 \cdot C_{\text{tool}}(t_t, t_{a_t}) + w_4 \cdot D_{\text{travel}}(\text{pos}_t, \text{pos}_{a_t})$$

Where:
* $C_{\text{machine}} = 1$ if $m_t \neq m_{a_t}$ (machine swap), else $0$.
* $C_{\text{setup}} = 1$ if $\theta_t \neq \theta_{a_t}$ (workpiece repositioning / rotation), else $0$.
* $C_{\text{tool}} = 1$ if $t_t \neq t_{a_t}$ (cutting tool swap), else $0$.
* $D_{\text{travel}} = \|\text{pos}_t - \text{pos}_{a_t}\|_2$ (Euclidean non-cutting air travel distance).
* $w_1, w_2, w_3, w_4$: Process knowledge weight coefficients ($w_{\text{machine}} > w_{\text{setup}} > w_{\text{tool}} > w_{\text{travel}}$).

**Reward Signal:**
$$R(s_t, a_t) = - C(s_t, a_t) + R_{\text{terminal}}$$
Where $R_{\text{terminal}} = +R_{\text{bonus}}$ if all $N$ steps are successfully completed, providing positive reinforcement upon sequence termination.

---

## 4. System Architecture & Module Design

```
+-------------------------------------------------------------------------+
|                          CAD & Feature Parser                           |
|  - Reads Part Feature Table (Rough/Semi-Finish/Finish Steps)            |
|  - Constructs Precedence DAG (Directed Acyclic Graph) via NetworkX      |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                  Gymnasium Manufacturing Environment                    |
|                      (`MachiningSequencingEnv`)                         |
|  - Tracks Machine, Orientation, Tool ID, Tool Position                  |
|  - Computes Dynamic Action Mask m(s_t) from Precedence DAG              |
|  - Computes Multi-Objective Cost (Machine + Setup + Tool + Travel)      |
+-------------------------------------------------------------------------+
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
+-----------------------------------+ +-----------------------------------+
|       DRL Agent (Masked PPO/DQN)  | |       Baseline Metaheuristics     |
| - MaskablePPO / Masked DQN Policy | | - Genetic Algorithm (GA)          |
| - Custom Actor-Critic Architecture| | - Simulated Annealing (SA)        |
| - Dynamic Action Logit Masking    | | - Priority Dispatching (SPT, FIFO)|
+-----------------------------------+ +-----------------------------------+
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
+-------------------------------------------------------------------------+
|                 Evaluation, Benchmarking & Visualization                |
|  - Solution Quality: Total Machining Cost Comparison                    |
|  - Efficiency: Inference Time (ms) vs GA Iterations                     |
|  - Visualization: Gantt Chart of Operations & Tool-Path Spatial Plots    |
+-------------------------------------------------------------------------+
```

---

## 5. Technology Stack & Software Requirements

| Category | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Primary development language |
| **RL Environment** | `Gymnasium` (v0.29+) | Standard RL environment API (`reset`, `step`, `action_masks`) |
| **RL Algorithms** | `sb3-contrib` / `Stable-Baselines3`, `PyTorch` | `MaskablePPO` / Action-Masked DQN implementation |
| **Graph & Combinatorics** | `NetworkX` | Feature precedence DAG construction and topological validation |
| **Data Processing** | `NumPy`, `Pandas` | Feature tables, distance matrices, state tensor manipulation |
| **Visualization** | `Matplotlib`, `Seaborn` | Learning curves, cost distributions, sequence Gantt charts |

---

## 6. Implementation Roadmap & Milestones

### Phase 1: Environment & Precedence Logic Implementation (Week 1–2)
* [ ] Define part representation schema (JSON / CSV) for multi-feature parts (Prismatic & Rotational parts).
* [ ] Build `MachiningSequencingEnv` adhering to Gymnasium interface.
* [ ] Implement strict precedence checks (datum-first, roughing-before-finishing, hole-making order).
* [ ] Implement invalid action masking function `action_masks(self)`.

### Phase 2: Classical Baselines Implementation (Week 2–3)
* [ ] Implement standard Genetic Algorithm (GA) tailored for permutation with precedence repair.
* [ ] Implement rule-based heuristics (Min-Tool-Change, Min-Setup-Change, FIFO).
* [ ] Validate baseline cost metrics and execution times on standard test parts (Part A: 10–15 steps, Part B: 25–40 steps).

### Phase 3: Knowledge-Guided DRL Training (Week 3–4)
* [ ] Configure `MaskablePPO` policy network with custom MLP / Feature Embedding layers.
* [ ] Tune hyperparameters (learning rate, entropy coefficient, batch size, reward scaling).
* [ ] Log training convergence, episode reward progression, and policy entropy.

### Phase 4: Comparative Benchmarking & Analysis (Week 4–5)
* [ ] Benchmark DRL vs GA vs SA vs Dispatching Rules on solution quality (Total Cost).
* [ ] Benchmark computational time (decision latency per step / full sequence generation).
* [ ] Test generalization on unseen part configurations.

### Phase 5: Documentation, Visuals & Presentation (Week 5–6)
* [ ] Generate operation Gantt charts showing grouped setups and tools.
* [ ] Generate convergence comparison curves.
* [ ] Final project report and presentation slide deck preparation.

---

## 7. Verification & Acceptance Criteria

1. **100% Precedence Constraint Compliance:** Every sequence generated by the trained DRL agent must strictly obey all geometric and technological precedence relations (0 invalid sequences).
2. **Superior or Comparable Solution Quality:** Total machining auxiliary cost ($C_{\text{total}}$) must be equal to or better than standard Genetic Algorithm solutions.
3. **Drastic Speedup in Inference:** Sequence generation runtime must be $\le 50\text{ ms}$ (at least $50\times$ faster than iterative GA convergence), enabling real-time adaptive CAPP.
4. **Reproducibility:** Self-contained code with seed control, modular classes, and automated benchmarking scripts.
