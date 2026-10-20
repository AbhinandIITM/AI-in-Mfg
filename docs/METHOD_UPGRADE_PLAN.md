# Method Upgrade Plan: Conditioned Policies, Domain Randomization, Transfer, and a Controller Teacher

**Project:** Energy-Aware DRL for Dynamic Cutting Parameter Optimization (ME6324, IIT Madras)
**Date:** October 20, 2026
**Status:** implemented, with the measured outcomes in section 10
**Predecessor:** `IMPROVEMENT_PLAN_AUDIT.md` (defect remediation, completed) and `docs/LITERATURE_COMPARISON.md`

---

## 1. Where the project stands

After the audit remediation the repository is functional and honest. The measured
baseline, from `results/RESULTS.md`:

- The Monte Carlo sweep runs 100 randomized scenarios (geometry, material and tool
  coefficients, initial flank wear) and reports 95% confidence intervals.
- The energy weight ablation trains 7 weights over 3 seeds and shows a clear,
  monotone effect of the energy weight on the operating point.
- The literature comparison quotes only numbers that exist in `docs/papers/` or
  in the artifacts of this repository.

Two measured facts drive this plan.

**Fact 1, the energy objective is not a trade-off in this model.** The energy of
a finished part is

$$E_{\text{part}} = k_c \cdot V + P_{\text{idle}} \cdot T_{\text{part}}$$

The cutting term is fixed by the volume that has to come off, and the idle term
falls exactly when the cycle time falls, which is also what raises the material
removal rate. Across the ablation grid the part energy falls by 30% and the
cycle time by 71% as the energy weight goes from 0.0 to 1.0, but the removal
rate rises as well. The two objectives point the same way, so the weight sweep
produces a monotone improvement rather than a Pareto frontier, and 21 separate
training runs buy a curve that one policy could produce directly.

**Fact 2, the policy is trained on one operating point and evaluated on many.**
The primary policy is trained on the nominal constants and then evaluated on
randomized scenarios. It is not conditioned on the scenario, so it has to
generalize to material and tool coefficients it has never seen, and the sweep
shows roughness violations in the double digits.

Both facts have the same remedy: put the context into the observation and
train across a distribution of contexts.

---

## 2. Methods considered, and the decision

| Method | Verdict | Reason |
| :--- | :--- | :--- |
| Multi-objective / conditioned RL (MORL style) | **Adopt** | Replaces the 21-run weight sweep with one policy that is conditioned on the energy weight, and gives the whole frontier from a single training run. |
| Domain randomization in the environment | **Adopt** | The policy currently trains on one operating point and is measured on many. Training across the same distribution it is measured on is the cheapest possible fix for the robustness claim. |
| Transfer learning | **Adopt** | Maps to a real deliverable (a new material or a different spindle limit), and is cheap because the observation space is fixed. |
| Receding-horizon controller as a teacher, behavioural cloning as warm start | **Adopt** | The current NSGA-II baseline is static, so it understates the classical approach. A receding-horizon solver is a genuinely strong expert, and cloning it is the standard way to spend simulation time efficiently. It also gives an honest "learning from a strong controller" section. |
| Continual learning in the classic sense | **Defer** | There is no task sequence yet. It becomes the right frame only if the target becomes non-stationarity, and then the conditioned policy with replay covers it. |
| Representation learning | **Reject for now** | The state is 8 physics-derived features, so there is nothing to learn a representation of. It becomes relevant only if raw sensor data (acoustic emission) enters the design, which is a different project. |
| Offline RL | **Reject** | No fixed dataset to be offline about. |
| Pure imitation learning | **Reject as the primary method** | Imitation is for when demonstrations are cheap and reinforcement learning is expensive. Here the simulator is the cheap data source and the reward is available, so imitation throws away information. It is used here as a warm start, not as a method. |
| Curiosity / intrinsic motivation | **Defer** | Exploration is not the bottleneck after the reward rebalance. Worth revisiting only if a task proves hard to learn. |

---

## 3. Phase 1: Domain randomization in the environment

### Design

`MillingDTEnv` already accepts per-scenario overrides for every physical
coefficient. What it does not do is draw them. The environment gains an
opt-in randomized mode:

- `MillingDTEnv(randomize=True)` samples the scenario at every `reset()`:
  workpiece geometry (`kappa_total`, `D`, `L_cut`), the material and tool
  coefficients (`C_wear`, `k1`, `k2`, `k3`, `alpha`, `beta`, `gamma_wear`),
  and the initial flank wear (`VB_0`).
- The sampling bands are the same bands the Monte Carlo sweep uses, so the
  policy is trained on the distribution it is measured on. The bands live in
  `config/physics_constants.py` so the environment and the sweep cannot drift
  apart.
- The derived scales (`MRR_max`, `E_ref`, `T_ref`) depend on the scenario, so
  they are recomputed whenever the scenario changes. The current code
  computes them once in `__init__`; that moves into a `_configure()` method
  called from `__init__` and from `reset()`.

### Expected outcome (hypothesis, to be measured)

Roughness violations in the Monte Carlo sweep fall, because the policy sees
scattered `k1` and pre-worn tools during training instead of meeting them for
the first time during evaluation.

### Files

- `config/physics_constants.py`: add `SCENARIO_BANDS`.
- `env/milling_env.py`: `_configure()`, randomized `reset()`, `randomize` flag.
- `agent/train_rl.py`: `env_kwargs` already reaches the environment, so the
  training script needs no change beyond passing `randomize=True`.
- `tests/test_env.py`: a test that two resets of a randomized environment draw
  different scenarios, and that a fixed environment does not.

---

## 4. Phase 2: The conditioned policy

### Design

The environment gains a second registration, `MillingDTEnv-w2-v1`, whose
observation carries the energy weight and the scenario context alongside the
eight physics features:

$$s_t = [VB, \kappa, n_{\text{prev}}, f_{\text{prev}}, a_{p,\text{prev}}, MRR_{\text{prev}}, Ra_{\text{prev}}, \text{broken}, w_2, \widetilde{C_{\text{wear}}}, \widetilde{k_1}]$$

The last three features are the conditioning variables. A single policy
trained over a sampled weight space and a sampled scenario distribution
answers, at runtime, "give me the most energy-efficient cut that still meets
the roughness limit for this material".

`MillingDTEnv-v1` keeps its 8-dimensional observation so the policies already
trained in `models/` stay loadable and the artifacts already committed stay
valid.

### Validation

The conditioned policy is the claim "one policy covers the frontier". The test
is whether its frontier reproduces the 21-policy frontier measured in
`results/ablation/ablation_summary.csv`. The two are compared in
`results/conditioned_vs_sweep.csv`.

### Files

- `env/milling_env.py`: observation composition, `condition_w2` flag.
- `env/__init__.py`: register `MillingDTEnv-w2-v1`.
- `agent/train_rl_conditioned.py`: training script for the conditioned policy.
- `agent/evaluate_conditioned.py`: sweeps the weight at runtime and writes the
  frontier.
- `tests/test_env.py`: observation dimension and content per registration.

---

## 5. Phase 3: Transfer learning across material and machine

### Design

A self-contained experiment, not a new training mode:

1. Train a source policy on the nominal material (source distribution).
2. Build a target task: a different material (shifted `k1`, `k2`, `C_wear`)
   and a different machine (a lower spindle ceiling, a different feed limit).
3. Measure three numbers on the target task:
   - the source policy zero-shot,
   - the source policy after fine-tuning for N episodes,
   - a policy trained on the target from scratch for the same episode budget.
4. Report the adaptation curve: target performance against episodes spent.

The comparison of 2 against 3 is the result that matters: it says how much of
the new task is covered by what the policy already learned.

### Files

- `agent/transfer_experiment.py`: the experiment, writing
  `results/transfer/transfer_results.csv` and an adaptation curve.
- `agent/summarize_results.py`: add the transfer table.

---

## 6. Phase 4: A receding-horizon controller, and cloning it

### Design

**The expert.** The existing `baselines/run_baselines.py` solves a static
multi-objective problem once for the whole part. A receding-horizon version
re-solves it at every pass with the current state (remaining stock, current
flank wear, current scenario) and takes the knee point of the resulting
Pareto set. This is a legitimate controller, and a much stronger classical
baseline than the static one.

**Cloning it.** The expert's trajectories are recorded as
`(observation, action)` pairs. A small supervised model with the same shape as
the PPO actor (256, 256, Tanh) is trained on them, and its weights are copied
into the PPO policy as the starting point for RL fine-tuning. The comparison
is against the same policy trained from scratch on the same step budget.

**The measurement.** Two numbers: the steps to reach a given episode reward
with and without the cloned initialization, and the final performance of each.

### Files

- `baselines/run_baselines.py`: add `solve_receding_horizon()`.
- `agent/clone_expert.py`: dataset collection, the supervised fit, and the
  weight transfer into a PPO policy.
- `agent/train_rl.py`: accept a policy to warm start from.
- `tests/`: the expert produces feasible actions on the nominal scenario.

---

## 7. Out of scope

- Representation learning, offline RL, curiosity, and classic continual
  learning, for the reasons in section 2. They are listed so that the
  decision is on record, and so that a future reader does not have to repeat
  the analysis.
- Any change to the physical models. The energy model stays as it is, and the
  limitation it imposes on the energy objective stays documented in
  `docs/LITERATURE_COMPARISON.md`.

---

## 8. Risks

| Risk | Mitigation |
| :--- | :--- |
| The conditioned policy needs more steps than a single-weight policy to cover the same weight range. | Train on a sampled weight space with the same budget used for the sweep, and compare against the measured sweep rather than against a guess. |
| Domain randomization slows learning, because every episode is different. | Measure against the current policy on the same budget before keeping the change. |
| The receding-horizon expert is slow enough to dominate the collection cost. | The expert solves a three-variable problem per pass; if it turns out to be the bottleneck, cache its solutions by state. |
| Fine-tuning numbers depend on the episode budget. | The budget is a parameter of the experiment and is reported with the result. |

---

## 9. Execution order

| Phase | Effort | Unblocks |
| :--- | :--- | :--- |
| 1. Domain randomization | 1.5 h | The robustness claim, phase 2 |
| 2. Conditioned policy | 3 h | Retires the 21-run sweep |
| 3. Transfer experiment | 2 h | A deployment story |
| 4. Controller teacher and cloning | 4 h | A fair classical comparison |

Each phase lands as its own commit with tests, and each produces an artifact
that the documentation quotes.

---

## 10. What was built, and what it measured

| Phase | Built in | Artifact | Outcome |
| :--- | :--- | :--- | :--- |
| 1. Domain randomization | `config/scenarios.py`, `env/milling_env.py` | `results/monte_carlo/eval_PPO_MC_v5.csv` | The finished surface stays inside the roughness limit in 100% of the 100 scenarios, the part completion rate is 98.0%, the flank wear limit is never exceeded. |
| 2. Conditioned policy | `env/__init__.py`, `agent/train_rl_conditioned.py`, `agent/evaluate_conditioned.py` | `results/conditioned/` | At a matched budget of 150,000 steps per model, one conditioned policy beats all 21 weight-specific policies: 12% to 22% more removal rate and 8% to 19% less energy per part, from one run instead of 21. |
| 3. Transfer experiment | `agent/transfer_experiment.py` | `results/transfer/` | Fine tuning the source policy for 50,000 steps removes no roughness violation (184.5 to 184.8), while a policy trained on the target from scratch for the same budget removes all of them (196.0). |
| 4. Controller teacher and cloning | `baselines/receding_horizon.py`, `agent/clone_expert.py` | `results/baseline_results_v5.csv`, `results/cloning/` | The receding-horizon controller scores +204.3 against -45 for every static baseline, and cloning it gives a small early advantage that is gone by 100,000 steps. |

Two results contradict what the plan expected, and both are reported as they
came out.

**The energy objective does not trade against throughput.** Section 1 predicted
that a conditioned policy would produce a frontier. It does, but the frontier
runs in the direction where both objectives improve, because the part energy
and the removal rate are tied together in this energy model. The physics is
the reason, not the method, and it is spelled out in
`docs/LITERATURE_COMPARISON.md`.

**Fine tuning is the wrong tool for a material change.** The plan assumed
that a pre-trained policy would adapt to a new material faster than one
trained from scratch. At a 50,000-step budget the opposite is true. The
reason the plan's own recommendation still holds is phase 1: a policy trained
across a distribution of materials needs no adaptation step at all.

