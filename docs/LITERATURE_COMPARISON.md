# Literature Comparison: DRL for Intelligent CNC Machining (2024 to 2025)

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

## 1. Sources

| Key | Citation | Local copy |
| :--- | :--- | :--- |
| **Mo et al. (2025)** | X. Mo, M. Xia, S. Mei, X. Hu, "Deep reinforcement learning-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation", *Journal of Manufacturing Processes* 156 (2025) 1006-1031. | `docs/papers/DRL-based optimization ...pdf` |
| **Wang et al. (2025a)** | P. Wang, Y. Cui, H. Tao, X. Xu, S. Yang, "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning", *Journal of Manufacturing Systems* 78 (2025) 124-152, doi 10.1016/j.jmsy.2024.11.013. | `docs/papers/Machining parameter optimization ...pdf` |
| **Wang et al. (2025b)** | P. Wang, Y. You, W. Liu, D. Ding, F. Xu, "Knowledge-guided deep reinforcement learning framework for feature machining step sequencing", *Int. J. Adv. Manuf. Technol.* 140 (2025) 5617-5636, doi 10.1007/s00170-025-16606-4. | `docs/papers/s00170-025-16606-4.pdf` |

---

## 2. What each work actually does

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

## 3. Published operating envelopes

Quoted from Mo et al. (2025), Table 5 and Section 3.4. The third column is
this repository, from `config/physics_constants.py`.

| Quantity | Mo et al. (2025) | This work |
| :--- | :--- | :--- |
| Cutting speed | 25 to 200 m/min | 314 to 1257 m/min at D = 50 mm, from 2000 to 8000 RPM |
| Feed per revolution | 0.05 to 0.3 mm/r | 0.05 to 0.35 mm/r |
| Depth of cut | 0.05 to 3 mm | 0.1 to 2.5 mm |
| Machining allowance | 3, 5, 6 or 10 mm | 5 mm nominal, 2 to 15 mm across the sweep |
| Surface roughness limit | Ra 3.2 um roughing, Ra 1.6 um finishing | Ra 1.6 um, 1.36 um with the 0.85 safety factor |
| Maximum spindle speed | 2000 RPM (TC-800), 4000 RPM for the new configuration | 8000 RPM |
| Tool wear limit | Separate roughing and finishing thresholds, plus a breakage state | Flank wear VB 0.3 mm, 0.255 mm with the safety factor |

---

## 4. What this work adds, and what it does not

**Adds**

1. Energy as an explicit objective. The reward carries a specific cutting
   energy term, `E = (P_idle + k_c * MRR) * t_cut`, and the energy weight `w2`
   is a first-class parameter of the environment rather than a by-product.
2. No sensor machine learning pipeline. Tool wear and surface roughness are
   propagated from the constants in `config/physics_constants.py`, so a run is
   reproducible from its seed and carries no training-data dependency.
3. Statistical robustness reporting. Every metric in the Monte Carlo sweep
   comes with a t-distribution 95% confidence interval over 100 randomized
   scenarios, covering workpiece geometry, material and tool coefficients, and
   the initial flank wear of the tool.
4. An explicit energy ablation. Seven energy weights from 0.0 to 1.0, three
   seeds each, all evaluated on the same scenario set, produce the empirical
   throughput-energy frontier in `results/plots/ablation_pareto_frontier.png`.
5. A conditioned policy. The observation carries the energy weight and the
   material context, and both are drawn per episode, so one policy covers the
   whole weight range instead of one policy per weight.
6. A receding-horizon controller as a classical baseline. The static NSGA-II
   baseline cannot react to the tool state, and all four static baselines end
   outside the roughness safety limit. The receding-horizon one re-solves at
   every pass and is the only classical policy that stays inside it.
7. A transfer experiment. A policy trained on one material is evaluated on a
   harder one, zero shot, then fine tuned, and compared against a policy
   trained on the new material from scratch for the same budget.

**Does not add**

1. No Soft Actor-Critic implementation. The repository trains and evaluates
   PPO only. An earlier version of this document reported SAC results with a
   16.4% energy reduction; no script here produced those numbers and the
   claim is withdrawn.
2. No tool replacement decision. The policy controls cutting parameters for a
   single tool life.
3. No real machine validation. The digital twin is calibrated against
   published models, not against measured cuts from a machining centre.
4. No learned surface roughness or tool condition estimator.

### What the energy objective can trade against, and what it cannot

The energy of a finished part in this digital twin is

$$E_{\text{part}} = k_c \cdot V + P_{\text{idle}} \cdot T_{\text{part}}$$

where $V$ is the volume that has to come off and $T_{\text{part}}$ is the cycle
time. The cutting term is fixed by the volume, so the only part of the energy
the policy controls is the idle term, and the idle term falls exactly when the
cycle time falls. The material removal rate is inversely proportional to the
same cycle time, $MRR = V / T_{\text{part}}$. Throughput and part energy
therefore point in the same direction in this model, and no weight on the
energy term can open a trade-off between them.

Two consequences follow, and both are visible in the ablation:

1. The binding constraint on the operating point is surface roughness and
   flank wear, not energy. A policy that pushes the feed rate up to raise the
   removal rate raises the roughness with it, and the roughness limit is what
   stops it.
2. The reward charges the energy of a pass, while the reported metric is the
   energy of a part. A part cut in many shallow passes draws less power per
   pass but spends more time on the idle machine, so the two quantities can
   move in opposite directions. The ablation reports both.

Opening a genuine throughput against energy trade-off needs an energy model
whose energy is not proportional to the removed volume, for example a spindle
power that grows with the cutting speed, together with a part level
accounting that charges the agent for the number of passes it creates. Both
are left as future work.


---

## 5. Measured results from this repository

Every table below is generated from the artifacts by
`python agent/summarize_results.py`, which writes the same content to
`results/RESULTS.md`.

### 5.1 Primary policy, Monte Carlo sweep over 100 scenarios

Artifact: `results/monte_carlo/eval_PPO_MC_v5.csv`

| Metric | Mean | 95% CI |
| :--- | ---: | :--- |
| Material removal rate (mm3/min) | 1325.7 | 1204.9 to 1446.5 |
| Energy per part (kJ) | 55.2 | 47.6 to 62.8 |
| Cycle time per part (s) | 16.7 | 14.4 to 18.9 |
| Max flank wear (mm) | 0.124 | 0.111 to 0.138 |
| Parts completed within spec | 98.0% | |
| Roughness violations, finished surface | 0.0% | 0.0% to 0.0% |
| Flank wear violations | 0.0% | 0.0% to 0.0% |
| Tool breakage | 2.0% | 0.0% to 4.7% |

The scenarios randomize the stock allowance, the tool diameter, the cutting
length, the wear and roughness coefficients, and the initial flank wear of the
tool. The policy is trained on scenarios drawn from the same tables, so it is
measured on the distribution it was trained on. The worst pass of an episode
exceeds the roughness limit in 40% of the scenarios, which is the roughing
pass; the reward constrains the finishing pass, which is the pass that sets
the surface of the part.

### 5.2 Energy weight ablation, 7 weights and 3 seeds

Artifact: `results/ablation/ablation_summary.csv`

| w2 | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Parts completed |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 835 +/- 73 | 79.1 +/- 1.3 | 62.6 +/- 2.4 | 0.99 |
| 0.4 | 925 +/- 55 | 66.4 +/- 2.1 | 36.9 +/- 4.1 | 1.00 |
| 0.8 | 925 +/- 65 | 63.3 +/- 1.0 | 31.0 +/- 2.1 | 0.99 |
| 1.0 | 952 +/- 23 | 62.4 +/- 0.5 | 29.1 +/- 1.0 | 1.00 |

### 5.3 One conditioned policy against 21 weight-specific policies

Artifacts: `results/conditioned/conditioned_frontier_150k.csv` and
`results/ablation/ablation_summary.csv`, both trained with 150,000 steps per
model.

| w2 | MRR, conditioned | MRR, per-weight | Energy, conditioned | Energy, per-weight |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1017.7 | 835.1 | 64.1 | 79.1 |
| 0.5 | 1058.8 | 944.4 | 59.6 | 64.7 |
| 1.0 | 1079.4 | 951.9 | 57.6 | 62.4 |

One training run covers the whole weight range and beats every weight-specific
policy on both objectives.

### 5.4 Classical baselines

Artifact: `results/baseline_results_v5.csv`

| Policy | MRR (mm3/min) | Energy per part (kJ) | Cycle time (s) | Mean reward |
| :--- | ---: | ---: | ---: | ---: |
| Constant high | 1099.6 | 13.9 | 4.3 | -45.2 |
| Constant safe | 1.6 | 1511.8 | 3000.0 | -140.4 |
| NSGA-II static | 733.0 | 15.0 | 6.4 | -45.5 |
| Rule based | 719.8 | 15.1 | 6.6 | -45.7 |
| Receding horizon | 981.7 | 14.2 | 4.8 | +204.3 |

The four static policies end outside the roughness safety limit, which the
reward penalizes with -50. The receding-horizon controller is the only
classical policy that stays inside it.

### 5.5 Transfer to a harder material

Artifact: `results/transfer/transfer_results.csv`

| Arm | Mean episode reward | Roughness violations | Parts completed |
| :--- | ---: | ---: | ---: |
| Source policy, zero shot | 184.5 | 100.0% | 93.3% |
| Source policy, fine tuned 50,000 steps | 184.8 | 100.0% | 93.3% |
| Trained on the target from scratch, 50,000 steps | 196.0 | 0.0% | 100.0% |


---

## 6. Reproducing the numbers above

```bash
# Primary policy
python agent/train_rl.py --algo PPO --steps 1000000 --seed 42

# Monte Carlo sweep with confidence intervals
python agent/evaluate.py

# Energy weight ablation, 7 weights x 3 seeds
python agent/train_rl_ablation.py --steps 100000 --scenarios 50 --slots 7

# Classical baselines
python baselines/run_baselines.py
```

---

## 7. Future work

1. Implement Soft Actor-Critic and re-run the energy weight ablation under it,
   to test whether the off-policy entropy regularizer finds a different
   operating point than PPO.
2. Deploy on a 3-axis machining centre with a power meter, and compare the
   measured energy per part against the digital twin prediction.
3. Extend the wear model from a single flank wear value to the separate
   roughing and finishing thresholds used by Mo et al. (2025).
