# Reinforcement Learning, PyTorch & AI Agent Training Skills Guide

This guide covers the specialized Machine Learning, Deep Reinforcement Learning (DRL), and Agent Training skills integrated into this workspace for both **Claude Code** and **Antigravity CLI (`agy`)**.

---

## Newly Integrated Skill Suites

| Suite / Source | Key Skills | Primary Purpose |
| :--- | :--- | :--- |
| **[Hugging Face Skills](https://github.com/huggingface/skills)** | `huggingface-llm-trainer`<br>`trl-training`<br>`huggingface-datasets`<br>`huggingface-trackio` | Cloud GPU training submission, LLM fine-tuning, TRL alignment (PPO, GRPO, DPO), Trackio monitoring |
| **[Mastering PyTorch RL & NLP](https://github.com/SpillwaveSolutions/mastering-pytorch-rl-nlp-agentic-skill)** | `mastering-pytorch-rl-nlp` | PyTorch 2.7+ / TorchRL (ClipPPOLoss, DQN), modern Gymnasium 1.0 (5-value step), AMP, torch.compile, PEFT/LoRA |
| **[ANFAIA Autonomous AI Training](https://github.com/ANFAIA/ANFAIA_Skills)** | `atraining-rl-recipe`<br>`atraining-rl-infra`<br>`atraining-openenv-environments`<br>`atraining-evaluation`<br>`atraining-agentic-rl-tito` | End-to-end RL training recipes, distributed rollout infrastructure, Gym environment formulation, benchmark evaluations |
| **[OpenPipe ART RL](https://github.com/OpenPipe/ART)** | `art-train-rl`<br>`train-rl` | Multi-turn RL workflows, programmatic rewards vs. RULER LLM judge scoring, asynchronous trajectory gathering |
| **[NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab)** | `isaaclab-training-rl-agents`<br>`isaaclab-train-rl-agents` | Physical simulation RL workflows, RSL-RL / SB3 / SKRL wiring, headless training, visual observation pipelines |

---

## Quick Command & Skill Mapping

### 1. Claude Code Commands

| Command | Backing Skills | What it triggers |
| :--- | :--- | :--- |
| **`/train-rl`** | `mastering-pytorch-rl-nlp`, `art-train-rl`, `atraining-rl-recipe`, `isaaclab-training-rl-agents` | Interactive RL pipeline wizard: Gym environment audit, reward formulation, policy network setup, and smoke test |
| **`/hf-train`** | `huggingface-llm-trainer`, `trl-training`, `huggingface-datasets` | Hugging Face cloud GPU job setup, LoRA/full fine-tuning, and Hub checkpoint push |
| **Direct Skill** | Mention skill name in prompt | Loads specific guidance from `.claude/skills/<skill-name>/SKILL.md` |

### 2. Antigravity CLI (`agy`) Invocations

In `agy`, skills can be called via their namespaced IDs or triggered naturally via prompt instructions:

```bash
# Explicit Skill Invocations:
agy /mastering-pytorch-rl-nlp "Audit agent/train_rl.py for modern Gymnasium compliance and TorchRL optimizations"
agy /art-train-rl "Design a rollout trajectory loop for the CNC milling environment"
agy /atraining-rl-recipe "Formulate PPO clipping, entropy schedule, and advantage normalization"
agy /huggingface-llm-trainer "Configure fine-tuning run with trackio experiment monitoring"
agy /isaaclab-training-rl-agents "Review headless simulation step performance and observation layout"
```

---

## Detailed Skill Profiles & Best Practices

### 1. Mastering PyTorch RL & NLP (`mastering-pytorch-rl-nlp`)

Provides expert guidelines on PyTorch 2.7+, TorchRL, and modern reinforcement learning paradigms:

- **Gymnasium 1.0 Compliance (Critical)**:
  - **Never** use `import gym` (deprecated).
  - **Always** use `import gymnasium as gym`.
  - **Step Return Signature**: `obs, reward, terminated, truncated, info = env.step(action)` (5 unpacked values).
- **Modern TorchRL Primitives**:
  - `GymEnv` wrappers for accelerated tensor conversions.
  - `SyncDataCollector` / `MultiprocessDataCollector` for parallel rollouts.
  - `ClipPPOLoss` with Generalized Advantage Estimation (`GAE`).
- **Performance Optimizations**:
  - `torch.compile(model)` for kernel fusion.
  - Automatic Mixed Precision (`torch.cuda.amp.autocast()` and `GradScaler`).

#### Example Usage in This Project:
```python
# Verifying modern Gymnasium 1.0 step loop in env/milling_dt_env.py
import gymnasium as gym

env = gym.make("MillingDTEnv-v1")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

### 2. OpenPipe ART RL Workflow (`art-train-rl` / `train-rl`)

An interactive wizard for designing robust reinforcement learning rollouts and scoring functions:

- **Step 1: Replayability**: Verifies whether episodes can be rolled out deterministically from identical seeds.
- **Step 2: Reward Architecture**:
  - **Programmatic Reward**: Recommended for deterministic physical metrics (e.g., surface roughness $Ra \le 1.6$, tool wear $VB \le 0.3$, energy consumption).
  - **RULER Judge**: Used when evaluating complex text or tool interaction trajectories via an LLM judge.
- **Step 3: Trajectory Gathering**:
  - Asynchronous trajectory gathering using `art.gather_trajectory_groups`.
  - Metric logging into `trajectory.reward` and `trajectory.metrics`.

#### Example Usage:
```text
Claude Code: /train-rl
Prompt: "Set up an asynchronous multi-pass rollout collector for the CNC milling environment"
```

---

### 3. ANFAIA Autonomous AI Training Suite

The ANFAIA suite provides deep recipes and infrastructure patterns:

- **`atraining-rl-recipe`**:
  - Policy gradient loss formulation (PPO, GRPO, DPO).
  - Advantage normalization and entropy regularization schedules to prevent early convergence to greedy local minima.
- **`atraining-openenv-environments`**:
  - Environment boundary validation.
  - Observation normalization ($z$-score vs. min-max).
  - Action space constraints and clipping.
- **`atraining-evaluation`**:
  - Monte Carlo evaluation sweeps.
  - Statistical confidence bounds ($p$-values, variance across seeds).

#### Example Usage in This Project:
```bash
agy "Use atraining-evaluation to inspect our 100-seed Monte Carlo sweep in agent/evaluate.py"
```

---

### 4. Hugging Face Skills (`huggingface-llm-trainer`, `trl-training`)

Provides cloud GPU training orchestration:

- **Hardware Auto-Sizing**: Selects optimal GPU instances (T4, A10G, A100, H100) based on model parameter count.
- **TRL Framework**: PPO, DPO, GRPO, and KTO alignment algorithms.
- **Trackio Telemetry**: Real-time loss and reward tracking dashboards without third-party dependencies.

#### Example Usage:
```text
Claude Code: /hf-train
Prompt: "Configure a TRL PPO alignment script using huggingface-llm-trainer"
```

---

### 5. NVIDIA IsaacLab Training RL Agents (`isaaclab-training-rl-agents`)

Provides patterns for physics-based simulation and digital twin RL workflows:

- **Config Decoupling**: Strictly separate framework configs (Stable-Baselines3, RSL-RL, SKRL, RL-Games).
- **Headless Execution**: Strip visual rendering pipelines during training iterations to maximize FPS.
- **Observation Pipeline**: Differentiate between state-based vectors (joint positions, feed rates, spindle speed) and sensory images.

---

## Complete Hands-On Examples for This Repository

### Scenario A: Optimizing PPO & SAC Machining Policies
```text
In Claude Code:
/train-rl Audit agent/train_rl.py and agent/run_experiments.py using mastering-pytorch-rl-nlp guidelines. Check for torch.compile opportunities and Gymnasium 5-tuple step returns.
```

### Scenario B: Formulating Multi-Objective Physical Rewards
```text
In Antigravity CLI:
agy "Using atraining-rl-recipe, review the reward balancing between energy consumption and tool flank wear in env/milling_dt_env.py"
```

### Scenario C: Launching Parallel Experiment Workers
```text
In Claude Code:
/plan Set up a 4-worker multiprocessing rollout manager in agent/run_experiments.py with proper CUDA device allocation
/build auto
/test tests/test_parallel.py
```
