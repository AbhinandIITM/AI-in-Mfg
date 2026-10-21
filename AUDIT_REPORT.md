# Code Audit Report: Digital Twin CNC Milling

**Date:** 2026-10-21
**Branch:** main
**Audited revision:** `20d9e83` (the line references below are against that revision)
**Disposition revision:** the commits that follow `01f5e35`, including the
consolidation onto the `MillingDT` package and the restructure of the run tree
**Project:** Digital Twin-Driven DRL for Energy-Aware CNC Milling Parameter Optimization (ME6324 - AI in Manufacturing, IIT Madras)

---

## 0. Status of this report

The report was written against revision `20d9e83` and every finding was then
checked against the code. This revision records the disposition of each
finding.

Two findings were marked P0, "affects scientific validity", and the report
recommended regenerating every result because of them. **Both were checked
with measurements, and neither holds.** The measurements are in section 12,
and they are reproducible with the scripts in section 13. No result was
regenerated on the strength of those two findings, because there was nothing
to regenerate for.

The findings that do hold were fixed, and the fixes are listed against each
item. The test suite grew from 21 tests to 34.

| | Count |
| :--- | ---: |
| Findings checked | 19 |
| Fixed | 9 |
| Dismissed with evidence | 6 |
| Partly valid, partly fixed | 3 |
| Still open | 1 |

---

## 1. Architecture & Structure (Strengths)

| Area | Assessment | Details |
|------|------------|---------|
| **Modularity** | ✅ Excellent | Clean separation: the `MillingDT` package holds the library (`env/`, `config/`, `agent/`, `baselines/`, `sim/`, `utils/`), `experiments/` holds the entry points, `viz/` holds the figures and `results/` holds the runs |
| **Physics Centralization** | ✅ Excellent | All equations in `MillingDT/config/physics.py`, used by env, baselines, sim, eval. No duplication. |
| **Constants Management** | ✅ Excellent | Single source of truth in `MillingDT/config/physics_constants.yaml`, loaded by `MillingDT/config/__init__.py`, with the literature citations |
| **Scenario Management** | ✅ Excellent | Shared `MillingDT/config/scenarios.py` for training and Monte Carlo evaluation, ensures no drift |
| **Experiment Orchestration** | ✅ Good | `experiments/run_pipeline.py` sequences the full pipeline; `experiments/run_experiments.py` handles parallel training with heartbeat monitoring |
| **Result Reproducibility** | ✅ Good | `experiments/summarize_results.py` rebuilds `results/RESULTS.md` from the committed artifacts; seeding via `MillingDT/utils/seeding.py` |

---

## 2. Critical Findings (Must Fix)

### 2.1 Energy Normalization Uses Reference Condition, Not Scenario Physics: **DISMISSED**

**Original claim:** `E_ref` is computed from the reference cutting condition
(fixed nominal parameters) rather than the current scenario's physics
coefficients, so the energy weight `w2` effectively changes magnitude from
scenario to scenario and the ablation is misleading.

**Why it does not hold.** `E_ref` is recomputed for every scenario, in
`_apply_scenario`, from that scenario's own physics. `reference_pass_energy`
takes the diameter, the cutting length and the energy coefficient table as
arguments, and the environment passes its own per-instance copies:

```python
self.E_ref = reference_pass_energy(self.D, self.L_cut, self.energy_cfg)
```

The reference cutting condition is the anchor of the speed dependent cutting
energy, not a stand-in for the scenario: it is where the calibrated `k_c0`
applies. The scenario coefficients that can change the energy, `k_c` and
`P_idle`, are read from the scenario's table.

Measured, on the same environment with one override at a time:

| Scenario | E_ref |
| :--- | ---: |
| nominal | 8.890 kJ |
| D = 100 mm | 14.781 kJ |
| L_cut = 200 mm | 17.781 kJ |
| k_c = 5.0 | 14.781 kJ |
| P_idle = 1000 W | 11.890 kJ |

The report also lists `C_wear` among the missing coefficients. `C_wear` has
no effect on the energy of a pass at all; it is a wear coefficient.

**Status:** no change needed. The suggestion in the report to "explicitly
document that normalization is to a fixed reference" was taken, in
`docs/ENVIRONMENT_SPEC.md` → `PROJECT.md` part 1, section 4: the reference
condition is the anchor of the speed dependence, and the scale follows the
scenario.

---

### 2.2 Action Scaling Allows Zero Feed Rate & Depth of Cut: **PARTLY FIXED**

The original observation is right and the original severity was not: the
action is clipped to `[-1, 1]` before the physical parameters are computed,
so `f_r` cannot reach zero and `t_cut = L_cut / (n f_r)` stays finite. The
report itself notes this and asks for documentation.

**Fixed:** `MillingDT/env/milling_env.py` now states that the action space is the
normalized box, that the policy output is unbounded, and that the clip is
the guard that keeps the physics inside the machine envelope.

---

### 2.3 VecNormalize Statistics Mismatch in Evaluation: **DISMISSED**

**Original claim:** the observation normalization statistics were computed on
training scenarios, the evaluation scenarios have different physics
coefficients, so the policy is fed off-distribution observations.

**Why it does not hold.** The evaluation scenarios are drawn from the same
tables as the training scenarios, from `config/scenarios.py`, which is the
point of the domain randomization: the policy is measured on the
distribution it was trained on. The observation features are also already
scale free across the scenarios, because the environment normalizes them
against the scenario's own limits: `VB` against `VB_max`, `kappa` against
`kappa_total`, the previous action against the machine envelope, `MRR`
against `MRR_max` and `Ra` against `Ra_max`.

Measured, on the trained policy, with the model's own VecNormalize
statistics applied to both sets:

| Feature | train mean | train std | sweep mean | sweep std |
| ---: | ---: | ---: | ---: | ---: |
| 0 (flank wear) | 0.057 | 0.970 | 0.009 | 0.915 |
| 1 (stock left) | -0.465 | 1.202 | -0.514 | 1.212 |
| 2 (previous speed) | 0.413 | 1.065 | 0.583 | 1.023 |
| 3 (previous feed) | 0.225 | 0.817 | 0.249 | 0.802 |
| 4 (previous depth) | 0.479 | 0.917 | 0.514 | 0.898 |
| 5 (previous removal rate) | 0.324 | 0.930 | 0.419 | 0.896 |
| 6 (previous roughness) | 0.098 | 0.786 | 0.124 | 0.777 |

Mean absolute difference of the means: **0.070**, against a standard
deviation of 1. The two sets are the same distribution. The three context
features are excluded from the table: the energy weight is a task parameter
rather than a scenario, so it is constant in both sets, and the two material
features are the context the policy is conditioned on.

The behaviour confirms it: over the 100 scenarios the policy completes 99.0%
of the parts, keeps the finished surface inside the roughness limit in every
one of them, and never exceeds the flank wear limit. A policy fed
off-distribution observations does not do that.

The three suggested fixes would each make the evaluation worse, and none was
applied.

**Status:** no change needed. The measurement was redone after the
consolidation onto the single 11-dimensional registration, and the conclusion
is unchanged.

---

### 2.4 Stochastic Breakage Uses `np_random` Without Seed Control: **FIXED**

The report's own verification confirmed the behaviour is correct and asked for
documentation.

**Fixed:** `MillingDT/env/milling_env.py` now states that the breakage draw comes from
the environment's own generator, which `reset(seed=...)` seeds, so a scenario
replayed with the same seed breaks in the same place.

---

## 3. Code Quality & Maintainability Issues

### 3.1 Duplicate Function Definitions: **FIXED**

The `generate_random_configs` wrapper in `experiments/evaluate.py` was real: it
re-exported the function from `MillingDT/config/scenarios.py` under the same
name, and the same function was imported twice in one file.

**Fixed:** the wrapper is gone. `experiments/evaluate.py` and
`experiments/train_rl_ablation.py` import it from
`MillingDT/config/scenarios.py` directly. `experiments/transfer_experiment.py`
already did.

### 3.2 Hardcoded Paths & Magic Strings: **FIXED**

**Fixed:** there is now one environment identifier, `MillingDT.env.ENV_ID`,
exported from `MillingDT/env/__init__.py` and used by the registration
itself. The paths are in `MillingDT/config/paths.yaml`, loaded by
`MillingDT/config/__init__.py`, which exposes `RESULTS_DIR`, `EVALUATION_DIR`,
`PLOTS_DIR`, `WORKERS_DIR` and the `run_dir`, `run_models`, `run_log`,
`run_plots`, `run_metrics` helpers. Every run writes into its own directory,
`results/run_<configuration>/`, and the files that repeated the literals now
import them. The consolidation that followed removed the second registration,
so the two identifier constants this finding asked for are now one.

### 3.3 Training Callback Redundancy: **NO CHANGE, AGREED**

The recommendation is to keep the heartbeat callback, and the report says so.
It is kept, and the reason is now in its docstring: the queue heartbeat is
what tells `experiments/run_experiments.py` that a worker is alive, and a
dropped heartbeat is indistinguishable from a hung one.

### 3.4 Inconsistent Error Handling in `run_experiments.py`: **PARTLY FIXED**

The valid part of this finding is the failure mode: if the log file cannot be
opened, the worker died before it trained anything, and the supervisor saw a
worker that exited without a heartbeat.

**Fixed:** the redirection is guarded. A worker that cannot open its log file
reports it and trains anyway, instead of dying silently. A worker that fails
for any other reason now exits with a non-zero code, and the supervisor checks
that the policy the job was supposed to train is on disk before it calls the
job a success.

**Still open:** the report's larger point, that `run_pipeline.py` and
`run_experiments.py` are two parallel execution strategies. They are, and
merging them is a design decision rather than a cleanup. See section 9.

### 3.5 Unused Imports & Dead Code: **PARTLY FIXED**

The list does not hold up. Every import named is used:

| Import | Used at |
| :--- | :--- |
| `time` in `experiments/train_rl.py` | the heartbeat timestamp |
| `configure` in `experiments/train_rl.py` | the CSV logger for the training curve |
| `stats` in `experiments/evaluate.py` | `stats.t.ppf` and `stats.norm.ppf` in `compute_statistics` |
| `Any` in `MillingDT/env/milling_env.py` | the type hints of `_apply_scenario` and `reset` |
| `ABLATION_WEIGHTS` in `MillingDT/sim/machining_sim.py` | snapping the requested energy weight to a trained one |
| `MillingDT.env` in `viz/plot_baselines_comparison.py` | the registration side effect |

**Fixed:** `typing.Tuple` in `MillingDT/env/milling_env.py` was genuinely
unused and is gone.

---

## 4. Testing Gaps

### 4.1 Missing Integration Tests: **MOSTLY FIXED**

**Added**, in `tests/test_integration.py`:

| Test | What it covers |
| :--- | :--- |
| `test_training_to_evaluation_roundtrip` | Train, save the policy and its normalization statistics, load them again, roll out over randomized scenarios and check the metrics are sane |
| `test_registered_environment_observation_carries_the_context` | The single registration exposes the weight and the material context, normalized |
| `test_baseline_comparison_writes_its_artifact` | The baseline script produces a row per policy and restores the committed artifact afterwards |

**Still open:** the ablation study has no end-to-end test. Running one weight
of the ablation takes minutes, which is too slow for the suite, and the
per-weight path is already covered by the roundtrip test.

### 4.2 Physics Correctness Tests: **FIXED**

**Added:** `test_reward_matches_an_independent_calculation` recomputes the
reward from `config/physics.py`, independently of the code that assembles
it, and asserts they agree to 1e-9. The reward is the sum of an objective
term, a progress term, the constraint penalties and the completion bonus, so
a change in any of them has to show up there.
`test_reward_penalizes_a_finishing_pass_out_of_spec` covers the finishing
branch, where the roughness penalty and the completion bonus apply.

**Coverage of `tests/test_env.py` now:** 19 tests, covering action clipping,
the truncated flag, the soft normalization, the constants sync, the physics
overrides, the info payload, the energy normalization, the Gymnasium checker,
the randomized scenario draw, the fixed scenario, the pinned values, the
scenario scales, the speed dependence of the cutting energy, the interior
minimum of the part energy, the shared physics, the shared scenario tables,
the golden value of the reward, and the roughness penalty.

---

## 5. Streamlit App (`app.py`): Specific Issues

### 5.1 Model Loading on Every Interaction: **DISMISSED**

The app already caches the policy. `app.py` line 112:

```python
@st.cache_resource(show_spinner="Loading Pre-Trained PPO Policy...")
```

`st.cache_resource` is the right decorator here, and the model is not
reloaded on a button press. The report's line references (180, 220, 260) point
at calls inside the cached helper.

### 5.2 Hardcoded Model Path: **FIXED**

**Fixed:** the default model directory is `MillingDT.config.PRIMARY_MODEL_DIR`,
and the parameter was already there for the callers that need a different
one.

### 5.3 G-Code Generator Uses Undefined Variables: **PARTLY FIXED**

The diagnosis is wrong: there is no undefined `tool_diameter` in the
function, and no `NameError` was reachable. The underlying observation is
right, though: the tool change line hardcoded `D50`, whatever diameter the
trajectory was actually cut with.

**Fixed:** `generate_cnc_gcode` takes a `tool_diameter` argument, the app
passes the diameter the run was cut with, and the tool change line now
reads `T01 M06 (D63 CORNER RADIUS END MILL)` for a 63 mm cutter.

---

## 6. Documentation Drift

### 6.1 README vs Implementation Mismatches: **PARTLY FIXED**

| README Claim | Reality | Action |
| :--- | :--- | :--- |
| "Observation space: 8-dimensional" | 11, the context is always present | The README now describes the 11-dimensional observation of the single registration |
| "Action space: continuous 3D" | Normalized `[-1, 1]`, the policy output is unbounded and clipped | The README now says so |

### 6.2 PROJECT.md Status Claims: **DISMISSED**

The report quotes a PROJECT.md claim of "18% mean reduction in energy per part
at 15% MRR reduction" and checks it against "w2=0: 82.8 kJ, w2=1: 62.5 kJ" and
"MRR 999.9 to 590.1".

Neither the quote nor the numbers are in the repository. The text does not
appear in `PROJECT.md`, 62.5 kJ is not in any artifact, and 590.1 matches
nothing.

The claim that was in `PROJECT.md` at the time of the audit, "across the
frontier the removal rate falls 18% and the energy of the part falls 8.5%",
did match `results/RESULTS.md` exactly: 1069 to 877 mm3/min is a fall of
18.0%, and 71.5 to 65.4 kJ is a fall of 8.5%. Both numbers were measured
again after the consolidation onto the single registration, and they moved:
the frontier now runs from 1038 to 951 mm3/min, a fall of 8.4%, against 72.7
to 65.8 kJ, a fall of 9.5%. The trade-off is symmetric now, which is what a
real Pareto frontier looks like. The document carries the new numbers.

**Status:** no change needed. `results/RESULTS.md` is regenerated from the
committed CSVs by `python -m experiments.summarize_results`, and the
documentation quotes those tables.

---

## 7. Security & Safety

### 7.1 No Security Vulnerabilities Found

- No user input reaches `eval()`/`exec()`
- No shell injection (subprocess uses fixed commands)
- No network exposure (local Streamlit only)
- File paths are constructed from `Path` objects, not user strings

### 7.2 Safety in Simulated Environment

The environment correctly enforces:

- Spindle speed limits (2000-8000 RPM)
- Feed rate limits (0.05-0.35 mm/r)
- Depth of cut limits (0.1-2.5 mm)
- Tool wear limit (VB ≤ 0.3 mm)
- Surface roughness limit (Ra ≤ 1.6 μm)

No code paths allow bypassing these constraints.

---

## 8. Performance Considerations

| Area | Observation | Recommendation | Status |
|------|-------------|----------------|--------|
| **Training Speed** | Single-threaded PyTorch (`torch.set_num_threads(1)`), correct for CPU training | Keep, it enables parallel workers | Kept |
| **VecEnv** | `make_vec_env` with **4** environments per process, not 16 | Correct as is | The report's figure of 16 is wrong |
| **Monte Carlo Eval** | Sequential episodes, 100 scenarios | Could be parallelized, not needed at this runtime | Accepted |
| **Streamlit** | The policy is cached | None | The report's row is wrong, it is already cached |
| **GPU** | PPO on an MLP is 2.5x slower on the GPU than on the CPU (318 against 812 steps/s measured on this machine) | Keep the CPU | Kept, now in the README compute notes |

---

## 9. Recommended Fix Priority Order, dispositioned

| Priority | Issue | Disposition |
|----------|-------|-------------|
| ~~P0~~ | Energy normalization uses the reference condition, not the scenario | Dismissed, section 2.1. No regeneration needed. |
| ~~P0~~ | VecNormalize statistics mismatch on randomized evaluation | Dismissed, section 2.3. No regeneration needed. |
| ~~P1~~ | Streamlit model reload on every interaction | Already cached, section 5.1 |
| ~~P1~~ | G-code generator bug | Fixed, section 5.3, the real defect was a hardcoded tool diameter |
| **P1** | Consolidate the two parallel execution strategies | **Open.** `run_pipeline.py` drives the pipeline through subprocesses, `run_experiments.py` supervises training workers through multiprocessing. They do different jobs, and merging them is a design decision with a real cost. |
| ~~P2~~ | Remove the duplicate `generate_random_configs` wrapper | Fixed, section 3.1 |
| ~~P2~~ | Centralize the environment identifiers and model paths | Fixed, section 3.2 |
| ~~P2~~ | Add a golden value reward test | Fixed, section 4.2 |
| ~~P3~~ | Clean up the unused imports | Fixed, section 3.5, only `typing.Tuple` was real |
| ~~P3~~ | Regenerate `RESULTS.md` and sync the PROJECT.md claims | Not needed, the claims already match the artifacts, section 6.2 |

---

## 10. Positive Observations (What Works Well)

1. **Physics correctness validated by tests**: `test_cutting_energy_grows_with_the_cutting_speed` and `test_part_energy_has_an_interior_minimum_in_speed` catch the key physics property that enables the energy/throughput trade-off, and `test_reward_matches_an_independent_calculation` now pins the reward itself.
2. **Domain randomization properly implemented**: `randomize=True` draws a new scenario per episode; the scales are recomputed in `_apply_scenario`; the same scenario tables are used for training and for the Monte Carlo sweep.
3. **One policy covers the full range**: the energy weight and the material context are in the observation of the single registration; one policy answers every weight at runtime. Against the 21 policies of the ablation study it is ahead by 1.4% to 5.6% of removal rate over the weights from 0.0 to 0.6 and behind at 0.8 and 1.0, by 9.0% and 21.6%, because the weight in the observation is a weaker signal than a dedicated policy where the trade-off is steepest.
4. **Receding-horizon baseline is a strong benchmark**: it re-solves at every pass with the current state, it uses the same physics, and the vectorized grid search is appropriate for a three dimensional action space.
5. **The transfer experiment measures the adaptation that matters**: a policy trained across a distribution of materials reaches the harder alloy with no adaptation at all, which is the argument for the context being in the observation.
6. **Comprehensive result artifacts**: CSVs for every experiment, plots generated from them, `RESULTS.md` rebuilt from the artifacts.
7. **Clean Git history**: conventional commits, descriptive messages, no merge commits in recent history.

---

## 11. What is left

1. **The one open item:** the two parallel execution strategies. Either merge
   them or document why both exist, in the same way the heartbeat callback
   now documents why it exists.
2. **The comparison against the classical controller, redone.** The original
   comparison was not a fair test, and three things were wrong with it. The
   reward did not count throughput: the objective term was worth about one
   point of reward per part against a completion bonus of 200, so the policy
   was rewarded for finishing a part at any cost. The controller was not a
   real baseline: it maximized only the objective term per pass, ignored the
   breakage probability, and committed to one feed for the whole part while
   the reward penalizes only the finishing pass. And the wear coefficient
   gave a tool life of about 870 parts, which left the wear constraint inert
   inside a part and the tool life irrelevant to a batch.
   All three are fixed. The reward scales now live in the configuration, the
   controller maximizes the expected episode reward over a full-part
   lookahead with a separate roughing and finishing feed, and the wear
   coefficient is anchored on the tool life that Wang et al. report.
   The measured outcome, including the boundary between the two methods, is
   in section 14.

---

## 12. How the two critical findings were checked

The scripts used for the measurements in sections 2.1 and 2.3 are reproduced
here so that anyone can repeat them.

```python
# 2.1, is E_ref computed from the scenario?
from MillingDT.env.milling_env import MillingDTEnv
print(MillingDTEnv().E_ref)              # 8.890 kJ, nominal
print(MillingDTEnv(D=100.0).E_ref)        # 14.781 kJ, the scenario's diameter
print(MillingDTEnv(k_c=5.0).E_ref)        # 14.781 kJ, the scenario's coefficient

# 2.3, are the evaluation observations off-distribution?
import numpy as np, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from MillingDT.config.scenarios import sample_scenario, generate_random_configs
from MillingDT.config import run_models, PRIMARY_RUN

model_dir = run_models(PRIMARY_RUN)
model = PPO.load(str(model_dir / "final_model.zip"), device="cpu")

def observations_for(scenarios, seed_base):
    rows = []
    for idx, scenario in enumerate(scenarios):
        e = gym.make("MillingDTEnv-v1", **scenario)
        e.reset(seed=seed_base + idx)
        vec = DummyVecEnv([lambda: e])
        vec = VecNormalize.load(str(model_dir / "vecnormalize.pkl"), vec)
        vec.training = False
        vec.norm_reward = False
        obs = vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec.step(action)
            rows.append(obs[0].copy())
            done = bool(done[0])
        return np.asarray(rows)

rng = np.random.default_rng(123)
train = observations_for([sample_scenario(rng) for _ in range(60)], 1000)
sweep = observations_for(list(generate_random_configs(60, seed=42).values()), 42)
print(np.abs(train.mean(0) - sweep.mean(0)).mean())   # 0.070
```

The script needs `import MillingDT.env` before `gym.make`, so that the
registration has happened.

---

## 13. Running the checks

```bash
# the test suite
python -m pytest tests/ -v

# the artifacts the documentation quotes
python -m experiments.summarize_results

# the sweep, which is deterministic and returns the same numbers every time
python -m experiments.evaluate

# the controller initialization experiment
python -m experiments.controller_experiment 1000000 2000

# the regime map, the boundary between the controller and the learned policy
python -m experiments.regime_map
```

---

## 14. The comparison against the classical controller, redone

The audit did not cover this comparison, and the review of it found three
defects, each of which had to be fixed before the comparison could mean
anything. The measurements below are on the same 100 randomized scenarios at
the same energy weight, w2 = 0.4, and they are in
`results/evaluation/robustness_results.csv`.

### 14.1 The reward did not count throughput

The reward was decomposed term by term over 30 scenarios. The objective term,
the throughput against the energy, was worth **1.2 points of reward per part**
for the controller and **-1.3** for the learned policy, against a completion
bonus of **200**. The whole trade-off the project is about was a rounding
error in the reward, and a policy that finished every part at any cost scored
well. The scales now live in `MillingDT/config/physics_constants.yaml` and the
objective term is worth about fifty points per part, a quarter of the value of
a part. That is a design decision and it is stated as one.

### 14.2 The classical controller was not a baseline

The controller maximized the objective term per pass and nothing else. It
ignored the breakage probability, the step penalty, the completion bonus and
the Pareto margin, it held the wear to `VB_max` where the reward penalizes
any pass above `VB_safe`, and it committed to one feed for the whole part
while the reward penalizes only the finishing pass. It is now a part-level
maximizer of the expected episode reward, with a separate roughing and
finishing feed and with the breakage risk priced. On the same scenarios it
went from 184.3 to 248.9.

### 14.3 The wear calibration was inert

`C_wear = 1.1e-4` gave a tool life of about 870 parts, so the wear
constraint did nothing inside a part and the tool life was irrelevant to a
batch. None of the three reference papers carries a Taylor wear model, so the
coefficient was never calibrated against one. It is now anchored on the tool
life that Wang et al. report, 40 parts cut with 5 identical tools, and gives
about 9 parts per tool.

### 14.4 What the comparison now says

| Regime | Arm | Reward | Std | MRR (mm3/min) | Energy (kJ) | Cycle (s) | Broken | Completed |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| The controller has every advantage | Receding-horizon | 248.9 | 79.5 | 1318.2 | 68.9 | 17.1 | 3% | 96% |
| The controller has every advantage | PPO | 233.6 | 79.7 | 1608.5 | 66.4 | 11.4 | 6% | 94% |
| The controller's material model is wrong | Receding-horizon | 63.5 | 117.1 | 1306.9 | 71.0 | 23.3 | 2% | 95% |
| The controller's material model is wrong | PPO | 226.9 | 88.3 | 1604.2 | 66.3 | 11.5 | 6% | 94% |
| The wear measurement is noisy | Receding-horizon | 181.3 | 185.9 | 1300.4 | 74.2 | 31.1 | 4% | 91% |
| The wear measurement is noisy | PPO trained with the noise | 213.6 | 87.3 | 1554.2 | 66.9 | 12.0 | 6% | 94% |

1. **Inside a perfect digital twin there is no reinforcement learning
   advantage, and that case is the left end of a boundary rather than a tie.**
   The controller computes the per-pass optimum of the same objective with the
   exact model, the exact state and the material in its hand, and it scores
   248.9 against the policy's 233.6. The gap is inside the noise: the
   standard error of the mean is about 8 for both arms, so a difference of 15
   has a t of about 1.4. Section 14.5 measures where the sign of the
   comparison changes.
2. **The learned policy is the better throughput operator.** It removes 22%
   more material per minute (1608 against 1318), finishes a part in a third
   less time (11.4 against 17.1 s) and spends 3.7% less energy. It pays for
   that with twice the tool breakage, 6% against 3%, and 4% of its finished
   surfaces above the roughness limit against 2%.
3. **The advantage appears where a plant actually is.** With a material the
   controller does not have a model for, its reward collapses by three
   quarters, from 248.9 to 63.5, because it does not slow down and its
   finished surfaces go out of specification. The learned policy holds at
   226.9, and it holds whether it is told the material (215.1) or not.
4. **The same holds for a noisy measurement.** With a 5% noise on the wear and
   the remaining stock the controller's reward falls to 181.3 and its spread
   more than doubles, to 185.9 against the policy's 87.3. The policy trained
   with the same noise keeps 213.6 and cuts the part in 12.0 s against the
   controller's 31.1 s.

So the honest conclusion is not that reinforcement learning beats a model
based controller. It is that the advantage of the learned policy appears in
the two places where the controller's information advantage disappears, and
that the boundary between the two is measured in section 14.5.

### 14.5 The regime map: where the boundary between the two methods is

The robustness experiment compares two situations. The regime map measures the
boundary between them, so the answer is a number a plant can act on. The
artifact is `results/evaluation/regime_map.csv` and the figure is
`results/plots/regime_map.png`, both written by
`python -m experiments.regime_map`.

The setup: the flank wear and the remaining stock are measured with an
instrument, not exactly, and the reading carries a Gaussian error of sigma, a
fraction of their range. The controller solves the next pass from that
reading; the policy reads the same noise on its observation.

**Nothing is tilted towards the policy, and two points are deliberate.**

- Both arms are the general ones, neither is trained for the noise. The policy
  that was trained with the noise exists and is reported as a third, clearly
  labelled arm, but it is not the headline comparison, because a controller
  can also be prepared for a noisy instrument by widening its safety margin,
  and that would be a design choice rather than a training one.
- The noise is tilted against the policy. The controller is given a noisy
  reading of the flank wear, the measurement it depends on, while the policy
  receives noise on the flank wear and on the remaining stock.

| Measurement noise | Receding-horizon | PPO | PPO trained for the noise | PPO ahead by |
| ---: | ---: | ---: | ---: | ---: |
| 0.0% | 248.9 | 233.6 | 219.6 | -15.3 |
| 1.0% | 242.3 | 233.4 | 216.9 | -8.9 |
| 2.0% | 231.5 | 233.1 | 214.3 | +1.7 |
| 3.0% | 211.5 | 230.0 | 214.1 | +18.5 |
| 5.0% | 181.3 | 226.8 | 213.6 | +45.5 |
| 7.5% | 179.1 | 223.2 | 207.9 | +44.1 |
| 10.0% | 171.5 | 222.0 | 206.5 | +50.6 |

1. **The crossover is at about 2% of the range of the wear measurement.**
   Below it the classical controller is the better operator, above it the
   learned policy is. That is the actionable form of the whole result.
2. **The two degrade differently.** The controller falls from 248.9 to 171.5,
   a 31% loss, because a noisy wear reading makes its plan violate the limits
   and break tools. The policy falls from 233.6 to 222.0, a 5% loss, because a
   perturbed observation perturbs its action slightly. A model-based
   controller has a cliff, a learned policy has a slope.
3. **Training for the noise does not pay.** The policy trained with the noise
   is below the general policy at every noise level, including the 5% it was
   trained for (213.6 against 226.8). The robustness of the general policy
   comes from being trained across a distribution of scenarios, and training
   against the noise only makes the policy more conservative.

Of the two tests, the noise is the fairer one, because the noise is a property
of the instrumentation and not of the method: the model error regime asks the
controller to be wrong about a material, while the noise asks it to work with
an imperfect instrument.

---

## 15. The reward is not neutral, and the comparison in the ideal case is biased

The reward credits the removal rate of every pass, which is the standard way
to express a rate in a reward that is paid per pass. It has a consequence
that has to be measured rather than argued about: the total credit grows
with the number of passes a plan takes, so a plan that cuts the same part in
more, shallower passes is paid more for it.

Measured on the same 100 randomized scenarios at w2 = 0.4, with the reward
terms accumulated separately by `experiments/reward_sensitivity.py`:

| | The controller | The policy |
| :--- | ---: | ---: |
| Passes per part | 5.85 | 4.00 |
| Objective credit collected | +0.413 | +0.326 |
| Mean reward | 248.9 | 233.6 |

The two plans remove the same 5 mm of stock. The controller takes more
passes, so it collects 27% more objective credit for the same material, and
at the objective scale of the configuration that is about nine of the fifteen
reward points that separate the two arms in the ideal case. **The ideal case
comparison is therefore biased against the learned policy, and the
defensible reading of it is that the two methods are comparable, with the
policy ahead on the physical metrics and behind on the tool breakage.** The
regime map of section 14.5 puts the ideal case at the left end of a boundary:
the crossover is at about 2% of the range of the wear measurement, and below
it the classical controller is the better operator.

Three things were checked before that conclusion was drawn:

1. **The objective scale is not the lever.** The controller wins at every
   scale from 5 to 4000, and the gap grows with it, from 7.1 points at 5 to
   352 points at 4000. The sign never flips. The sensitivity table is in
   `results/evaluation/reward_sensitivity.csv`.
2. **The alternative was implemented and rejected.** Crediting the material
   of the part against the energy of the part removes the pass count from
   the accounting, and under that form the two arms are level at the current
   scale and the policy is ahead above it. It is not the right fix, because
   the material a part gives up is fixed by its stock: the volume of a part
   is `pi * D * kappa * L_cut / 1000`, independent of every cutting
   parameter. The throughput weight would then be inert and the reward would
   be an energy objective with a throughput weight that does nothing, and
   the weight sweep of the project would stop measuring what it claims to
   measure.
3. **What a correct formulation would need.** The removal rate of a part is
   the ratio of a fixed volume to the part's time, and a rate cannot be
   accumulated over a part without becoming a volume. A reward that expresses
   it has to either price the cycle time directly, or compare the part's
   average rate to a reference at the end of the part. Both change the
   meaning of the weights, and the second makes the objective sparse and the
   credit assignment poor. Neither was adopted here, and the reward as
   shipped carries the artifact instead, documented rather than hidden.

This is the open question of the work, and it belongs in the future work of
the report: the comparison against a model-based controller is sensitive to
how the reward pays for a rate, and the sensitivity has to be published
alongside the result.

---

**End of Audit Report**
