# Code Audit Report: Digital Twin CNC Milling

**Date:** 2026-10-20
**Branch:** main
**Audited revision:** `20d9e83` (the line references below are against that revision)
**Disposition revision:** `01f5e35` and the commits that follow it
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
item. The test suite grew from 21 tests to 26.

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
| **Modularity** | ✅ Excellent | Clean separation: `config/` (physics, scenarios, constants, paths), `env/` (Gymnasium env), `agent/` (training/eval), `baselines/`, `viz/`, `sim/`, `utils/` |
| **Physics Centralization** | ✅ Excellent | All equations in `config/physics.py` — used by env, baselines, sim, eval. No duplication. |
| **Constants Management** | ✅ Excellent | Single source of truth in `config/physics_constants.py` with literature citations |
| **Scenario Management** | ✅ Excellent | Shared `config/scenarios.py` for training and Monte Carlo evaluation — ensures no drift |
| **Experiment Orchestration** | ✅ Good | `run_pipeline.py` sequences the full pipeline; `run_experiments.py` handles parallel training with heartbeat monitoring |
| **Result Reproducibility** | ✅ Good | `summarize_results.py` rebuilds `RESULTS.md` from committed artifacts; seeding via `utils/seeding.py` |

---

## 2. Critical Findings (Must Fix)

### 2.1 Energy Normalization Uses Reference Condition, Not Scenario Physics — **DISMISSED**

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

### 2.2 Action Scaling Allows Zero Feed Rate & Depth of Cut — **PARTLY FIXED**

The original observation is right and the original severity was not: the
action is clipped to `[-1, 1]` before the physical parameters are computed,
so `f_r` cannot reach zero and `t_cut = L_cut / (n f_r)` stays finite. The
report itself notes this and asks for documentation.

**Fixed:** `env/milling_env.py` now states that the action space is the
normalized box, that the policy output is unbounded, and that the clip is
the guard that keeps the physics inside the machine envelope.

---

### 2.3 VecNormalize Statistics Mismatch in Evaluation — **DISMISSED**

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
| 0 (flank wear) | 0.076 | 0.969 | 0.015 | 0.917 |
| 1 (stock left) | -0.491 | 1.216 | -0.534 | 1.229 |
| 2 (previous speed) | 0.463 | 1.044 | 0.526 | 1.023 |
| 3 (previous feed) | 0.278 | 0.890 | 0.314 | 0.889 |
| 4 (previous depth) | 0.487 | 0.909 | 0.512 | 0.895 |
| 5 (previous removal rate) | 0.346 | 0.942 | 0.406 | 0.940 |
| 6 (previous roughness) | 0.201 | 0.914 | 0.239 | 0.915 |

Mean absolute difference of the means: **0.041**, against a standard
deviation of 1. The two sets are the same distribution.

The behaviour confirms it: over the 100 scenarios the policy completes 99.0%
of the parts, keeps the finished surface inside the roughness limit in every
one of them, and never exceeds the flank wear limit. A policy fed
off-distribution observations does not do that.

The three suggested fixes would each make the evaluation worse, and none was
applied.

**Status:** no change needed.

---

### 2.4 Stochastic Breakage Uses `np_random` Without Seed Control — **FIXED**

The report's own verification confirmed the behaviour is correct and asked for
documentation.

**Fixed:** `env/milling_env.py` now states that the breakage draw comes from
the environment's own generator, which `reset(seed=...)` seeds, so a scenario
replayed with the same seed breaks in the same place.

---

## 3. Code Quality & Maintainability Issues

### 3.1 Duplicate Function Definitions — **FIXED**

The `generate_random_configs` wrapper in `agent/evaluate.py` was real: it
re-exported the function from `config/scenarios.py` under the same name, and
the same function was imported twice in one file.

**Fixed:** the wrapper is gone. `agent/evaluate.py` and
`agent/train_rl_ablation.py` import it from `config/scenarios.py` directly.
`agent/transfer_experiment.py` already did.

### 3.2 Hardcoded Paths & Magic Strings — **FIXED**

**Fixed:** the environment identifiers are now `env.ENV_ID` and
`env.CONDITIONED_ENV_ID`, exported from `env/__init__.py` and used by the
registration itself. The model and result directories are in the new
`config/paths.py` (`MODELS_DIR`, `RESULTS_DIR`, `PLOTS_DIR`, `CURVES_DIR`,
`WORKERS_DIR`, `PRIMARY_MODEL_DIR`, `CONDITIONED_MODEL_DIR`), and the eight
files that repeated the literals now import them.

### 3.3 Training Callback Redundancy — **NO CHANGE, AGREED**

The recommendation is to keep the heartbeat callback, and the report says so.
It is kept, and the reason is now in its docstring: the queue heartbeat is
what tells `run_experiments.py` that a worker is alive, and a dropped
heartbeat is indistinguishable from a hung one.

### 3.4 Inconsistent Error Handling in `run_experiments.py` — **PARTLY FIXED**

The valid part of this finding is the failure mode: if the log file cannot be
opened, the worker died before it trained anything, and the supervisor saw a
worker that exited without a heartbeat.

**Fixed:** the redirection is guarded. A worker that cannot open its log file
reports it and trains anyway, instead of dying silently.

**Still open:** the report's larger point, that `run_pipeline.py` and
`run_experiments.py` are two parallel execution strategies. They are, and
merging them is a design decision rather than a cleanup. See section 9.

### 3.5 Unused Imports & Dead Code — **PARTLY FIXED**

The list does not hold up. Every import named is used:

| Import | Used at |
| :--- | :--- |
| `time` in `agent/train_rl.py` | the heartbeat timestamp |
| `configure` in `agent/train_rl.py` | the CSV logger for the training curve |
| `stats` in `agent/evaluate.py` | `stats.t.ppf` and `stats.norm.ppf` in `compute_statistics` |
| `Any` in `env/milling_env.py` | the type hints of `_apply_scenario` and `reset` |
| `ABLATION_WEIGHTS` in `sim/machining_sim.py` | snapping the requested energy weight to a trained one |
| `env` in `viz/plot_baselines_comparison.py` | the registration side effect |

**Fixed:** `typing.Tuple` in `env/milling_env.py` was genuinely unused and is
gone. An AST scan over the project now reports no unused imports.

---

## 4. Testing Gaps

### 4.1 Missing Integration Tests — **MOSTLY FIXED**

**Added**, in `tests/test_integration.py`:

| Test | What it covers |
| :--- | :--- |
| `test_training_to_evaluation_roundtrip` | Train, save the policy and its normalization statistics, load them again, roll out over randomized scenarios and check the metrics are sane |
| `test_conditioned_environment_observation_carries_the_context` | The conditioned registration exposes the weight and the material context, normalized |
| `test_baseline_comparison_writes_its_artifact` | The baseline script produces a row per policy and restores the committed artifact afterwards |

**Still open:** the ablation study has no end-to-end test. Running one weight
of the ablation takes minutes, which is too slow for the suite, and the
per-weight path is already covered by the roundtrip test.

### 4.2 Physics Correctness Tests — **FIXED**

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

## 5. Streamlit App (`app.py`) — Specific Issues

### 5.1 Model Loading on Every Interaction — **DISMISSED**

The app already caches the policy. `app.py` line 112:

```python
@st.cache_resource(show_spinner="Loading Pre-Trained PPO Policy...")
```

`st.cache_resource` is the right decorator here, and the model is not
reloaded on a button press. The report's line references (180, 220, 260) point
at calls inside the cached helper.

### 5.2 Hardcoded Model Path — **FIXED**

**Fixed:** the default model directory is `config.paths.PRIMARY_MODEL_DIR`,
and the parameter was already there for the callers that need a different
one.

### 5.3 G-Code Generator Uses Undefined Variables — **PARTLY FIXED**

The diagnosis is wrong: there is no undefined `tool_diameter` in the
function, and no `NameError` was reachable. The underlying observation is
right, though: the tool change line hardcoded `D50`, whatever diameter the
trajectory was actually cut with.

**Fixed:** `generate_cnc_gcode` takes a `tool_diameter` argument, the app
passes the diameter the run was cut with, and the tool change line now
reads `T01 M06 (D63 CORNER RADIUS END MILL)` for a 63 mm cutter.

---

## 6. Documentation Drift

### 6.1 README vs Implementation Mismatches — **PARTLY FIXED**

| README Claim | Reality | Action |
| :--- | :--- | :--- |
| "Observation space: 8-dimensional" | 8 for `MillingDTEnv-v1`, 11 for `MillingDTEnv-w2-v1` | The README now points at the conditioned registration and its extra features |
| "Action space: continuous 3D" | Normalized `[-1, 1]`, the policy output is unbounded and clipped | The README now says so |

### 6.2 PROJECT.md Status Claims — **DISMISSED**

The report quotes a PROJECT.md claim of "18% mean reduction in energy per part
at 15% MRR reduction" and checks it against "w2=0: 82.8 kJ, w2=1: 62.5 kJ" and
"MRR 999.9 to 590.1".

Neither the quote nor the numbers are in the repository. The text does not
appear in `PROJECT.md`, 62.5 kJ is not in any artifact, and 590.1 matches
nothing. The current claim in `PROJECT.md` part 1, section 9, and in part 3,
section 5.2, is "across the frontier the removal rate falls 18% and the
energy of the part falls 8.5%", and it matches
`results/RESULTS.md` exactly: 1069 to 877 mm3/min is a fall of 18.0%, and
71.5 to 65.4 kJ is a fall of 8.5%.

**Status:** no change needed. `results/RESULTS.md` is regenerated from the
committed CSVs by `python agent/summarize_results.py`, and the documentation
quotes those tables.

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

## 9. Recommended Fix Priority Order — dispositioned

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

1. **Physics correctness validated by tests** — `test_cutting_energy_grows_with_the_cutting_speed` and `test_part_energy_has_an_interior_minimum_in_speed` catch the key physics property that enables the energy/throughput trade-off, and `test_reward_matches_an_independent_calculation` now pins the reward itself.
2. **Domain randomization properly implemented** — `randomize=True` draws a new scenario per episode; the scales are recomputed in `_apply_scenario`; the same scenario tables are used for training and for the Monte Carlo sweep.
3. **Conditioned policy design is sound** — the energy weight and the material context are in the observation; a single policy covers the full range; the transfer experiment measures the adaptation.
4. **Receding-horizon baseline is a strong benchmark** — it re-solves at every pass with the current state, it uses the same physics, and the vectorized grid search is appropriate for a three dimensional action space.
5. **Comprehensive result artifacts** — CSVs for every experiment, plots generated from them, `RESULTS.md` rebuilt from the artifacts.
6. **Clean Git history** — conventional commits, descriptive messages, no merge commits in recent history.

---

## 11. What is left

1. **The one open item:** the two parallel execution strategies. Either merge
   them or document why both exist, in the same way the heartbeat callback
   now documents why it exists.
2. **The learned policy still does not beat the classical controller.** On
   the same 50 episodes the policy scores 204.1 and the controller 203.5, and
   the controller is ahead on the removal rate and the cycle time. The next
   experiment is to start the policy from the controller and fine-tune from
   there. This is a finding about the work, not about the code, and it is
   recorded in `README.md` and `PROJECT.md`.

---

## 12. How the two critical findings were checked

The scripts used for the measurements in sections 2.1 and 2.3 are reproduced
here so that anyone can repeat them.

```python
# 2.1, is E_ref computed from the scenario?
from env.milling_env import MillingDTEnv
print(MillingDTEnv().E_ref)              # 8.890 kJ, nominal
print(MillingDTEnv(D=100.0).E_ref)        # 14.781 kJ, the scenario's diameter
print(MillingDTEnv(k_c=5.0).E_ref)        # 14.781 kJ, the scenario's coefficient

# 2.3, are the evaluation observations off-distribution?
import numpy as np, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config.scenarios import sample_scenario, generate_random_configs

model_dir = PROJECT_ROOT / "models" / "PPO_seed42"
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
print(np.abs(train.mean(0) - sweep.mean(0)).mean())   # 0.041
```

---

## 13. Running the checks

```bash
# the test suite
python -m pytest tests/ -v

# the artifacts the documentation quotes
python agent/summarize_results.py

# the sweep, which is deterministic and returns the same numbers every time
python agent/evaluate.py
```

---

**End of Audit Report**
