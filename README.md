# Digital Twin-Driven DRL for Energy-Aware CNC Milling Parameter Optimization

**Course:** ME6324 — AI in Manufacturing (IIT Madras)

## Overview
This project applies Deep Reinforcement Learning (DRL) to solve the dynamic cutting parameter optimization problem in CNC milling. Using a Python-based Digital Twin, an RL agent dynamically adjusts spindle speed ($n$), feed rate ($f_r$), and depth of cut ($a_p$) across multi-pass machining operations to achieve a multi-objective goal:
* Maximize throughput (Material Removal Rate)
* Minimize energy consumption (Green Manufacturing)
* Guarantee surface quality ($Ra \le 1.6 \mu m$)
* Prevent catastrophic tool breakage ($VB \le 0.3$ mm)

## Key Results & Findings
We trained and evaluated **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** algorithms utilizing reward normalization, entropy tuning, and expanded neural architectures (256x256).

1. **SAC Dominance (The "Eco-Zone")**: SAC discovered a highly energy-efficient strategy for roughing that perfectly balanced tool wear and power consumption, minimizing overall cost.
2. **Finishing Consistency**: Both algorithms independently learned the optimal physics-based finishing strategy: dropping the feed rate to the minimum safe boundary (~0.05 mm/r) and spiking the spindle speed to guarantee a mirror finish.
3. **Algorithm Robustness**: Through massive Monte Carlo statistical testing across hundreds of randomized scenarios, SAC proved to be incredibly robust, safely adhering to all engineering constraints across the vast majority of runs.

## Repository Structure
* `env/`: Gymnasium environment (`MillingDTEnv-v1`) incorporating empirical physics equations.
* `agent/`: Core Deep RL implementation.
  * `train_rl.py`: Single-run training script for PPO and SAC.
  * `run_experiments.py`: Parallel worker manager for concurrent training.
  * `evaluate.py`: Runs the **Massive Monte Carlo Random Sweep** evaluation and exports trajectory CSVs.
* `baselines/`: Implementation of classical/static baselines including NSGA-II and rule-based decay.
* `viz/`: Data visualization scripts.
  * `plot_eval.py`: Generates constraint KDE distributions and failure landscape scatters.
  * `plot_results.py`: Generates comparative bar charts for action strategies.
* `logs/`: Tensorboard training logs and Monte Carlo evaluation CSV datasets.
* `models/`: Saved `.zip` DRL policies and `.pkl` normalization statistics.
* `plots/`: Output directory for generated visualizations.
* `ppt/`: Contains the LaTeX Beamer code for the project presentation.
* `docs/`: Reference papers, project planning documents, and evaluation results.

## How to Run

### 1. Train Models (Parallel Execution)
To train PPO and SAC concurrently (utilizing the slot-worker pattern):
```bash
python agent/run_experiments.py
```
*(Alternatively, train a single model using `python agent/train_rl.py --algo SAC`)*

### 2. Evaluate Models (Monte Carlo Sweep)
To test the trained models on the Digital Twin across 100 randomized physical configurations and export trajectory logs to `logs/monte_carlo/`:
```bash
python agent/evaluate.py
```

### 3. Generate Statistical Plots
To parse the Monte Carlo evaluation CSVs and generate KDE distribution and Failure Landscape plots:
```bash
python viz/plot_eval.py
```
*(Plots are exported to the `plots/` directory).*

## References
* **"DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"**, *J. Manuf. Processes 156* — MDP formulation, Taylor-extended tool wear environment.
* **"Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning"**, *J. Manuf. Systems 78* — Multi-task estimation and surface roughness surrogate models.
