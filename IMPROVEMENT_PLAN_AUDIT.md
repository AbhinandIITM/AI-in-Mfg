# Improvement Plan Implementation Audit

**Plan audited:** `IMPROVEMENT_PLAN.md` (dated October 8, 2026)
**Audit date:** October 20, 2026
**Repository commit:** `9a8bf6e` ("feat: Implement CNC milling simulation engine and material removal calculator")
**Stack under audit:** Python 3.12.13, gymnasium 1.3.0, stable-baselines3 2.9.0

---

## 1. Verdict

Roughly 90 percent of the plan's code changes exist on disk. Three of them do not work, and the plan's headline scientific claim (SAC) has no implementation anywhere in the repository.

| Measure | Result |
|---|---|
| Plan items with code present | 9 of 9 |
| Plan items that function as specified | 6 of 9 |
| Plan success criteria met | 5 of 7 |
| Test suite | 6 passed, 0 warnings (`pytest tests/ -v`) |
| `gymnasium` environment checker | Passes clean |

The three non-functional items are not cosmetic. They are the queue heartbeat that terminates parallel training runs (Section 4.1), the Monte Carlo sweep that does not randomize what it claims to randomize (Section 4.2), and the energy ablation whose committed results show no trade-off at all (Section 4.4). A fourth issue sits outside the plan entirely but blocks reproducibility: the core environment file has never been committed to git (Section 4.5).

---

## 2. Method

Every source file listed in the plan was read in full. The following checks were executed rather than inferred:

| Check | Command | Result |
|---|---|---|
| Unit and integration tests | `python -m pytest tests/ -v` | 6 passed in 8.50s |
| Gymnasium API compliance | `gymnasium.utils.env_checker.check_env` on `MillingDTEnv` | Passes, no warnings from the environment |
| Queue heartbeat | Direct invocation of `ProgressQueueCallback._on_step` with a populated `locals` dict | No message reaches the queue |
| Config override handling | `gym.make("MillingDTEnv-v1", C_wear=0.05, k1=14.0, ...)` | All three overrides silently ignored |
| Env info payload | Inspection of the `info` dict returned by `step()` | No `T_total` key |
| Ablation policy divergence | Deterministic rollout of all 8 ablation models | 3 models identical, 5 models distinct |
| Version control coverage | `git ls-files`, `git check-ignore -v`, `git show HEAD:<path>` | `env/` absent from HEAD |
| SAC presence | `grep -r "SAC" --include=*.py` | 0 matches |

All numbers quoted below come from these runs or from committed CSV artifacts.

---

## 3. Item-by-item scorecard

| # | Plan item | Status | Location |
|---|---|---|---|
| 1.1 | `truncated` flag | Implemented | `env/milling_env.py:77`, `:172` |
| 1.2 | Action clipping | Implemented | `env/milling_env.py:125` |
| 1.3 | Soft tanh normalization | Implemented | `env/milling_env.py:103-108`, used at `:112-118` |
| 1.4 | Intermediate rewards | Partial | `env/milling_env.py:185-209`. Kappa progress bonus, terminal bonus, and penalties are present. The per-pass "stay in spec" bonus from the plan was replaced by a penalty design, which is a reasonable deviation. |
| 2.1 | Queue-based progress polling | **Non-functional** | Section 4.1 |
| 2.2 | Central environment registration | Implemented | `env/__init__.py`; `agent/train_rl.py:24` and `agent/evaluate.py:17` both `import env`. No duplicate `register()` calls remain. |
| 2.3 | Global seeding | Implemented | `utils/seeding.py`; used at `agent/train_rl.py:45`, `agent/evaluate.py:81`, `tests/test_training.py:29,58` |
| 3.1 | Monte Carlo 95 percent CI | Partial | CI math is correct at `agent/evaluate.py:38-78`. Two defects degrade the output (Sections 4.2 and 4.3). |
| 3.2 | Visualization DRY refactor | Implemented | `viz/plotting_utils.py`; consumed by `viz/plot_eval.py:11`, `viz/plot_results.py:12`, `viz/plot_baselines_comparison.py:20` |
| 3.3 | Centralized physics constants | Implemented | `config/physics_constants.py`; imported by `env/milling_env.py:21` and `baselines/run_baselines.py:28`. Wear, roughness, and energy formulas verified identical between the two consumers. |
| 4.1 | Energy weight ablation | **Degenerate output** | Section 4.4 |
| 4.2 | Literature comparison | Implemented, unsupported claims | Section 5 |
| Part 6 | Unit and integration tests | Implemented | `tests/test_env.py` (4 tests), `tests/test_training.py` (2 tests) |
| Part 7 | Success criteria | 5 of 7 met | Section 6 |

---

## 4. Functional defects

### 4.1 The queue heartbeat never fires, and it kills parallel training runs (P0)

**Files:** `agent/train_rl.py:37`, `agent/train_rl.py:122`, `agent/run_experiments.py:98`

`import numpy as np` sits inside the `if __name__ == "__main__":` block at `agent/train_rl.py:121-122`. When `agent.run_experiments` imports the module to spawn a worker, that import never executes, so `np` does not exist in the module namespace.

The progress callback at `agent/train_rl.py:37` evaluates `np.mean(...)` while building the queue payload. The `NameError` is raised inside a `try` block whose `except` clause is a bare `except: pass`, so it is swallowed and the payload is never sent.

Probe result with a realistic `locals` payload:

```
msg after real-path locals: []
```

Consequence: `agent/run_experiments.py:98` never receives a heartbeat, so `last_update[slot_id]` is never refreshed. After 300 seconds the stall detector terminates the worker. Every training run longer than 5 minutes is killed, and the default configuration in `agent/run_experiments.py:129` requests 200,000 steps.

The `models/ablation_w2_0_2`, `w2_0_4`, `w2_0_6`, and `w2_0_8` checkpoints on disk were therefore produced by some other route, not by the script in its current state.

**Fix:** move `import numpy as np` to module scope in `agent/train_rl.py`, and replace the bare `except: pass` with one that logs the exception.

### 4.2 The Monte Carlo sweep does not randomize the physics it claims to (P1)

**Files:** `agent/evaluate.py:29-35`, `env/milling_env.py:26-63`

`generate_random_configs` draws random values for `C_wear`, `k1`, and `k2` and passes them as environment kwargs. `MillingDTEnv.__init__` accepts only `n_min`, `n_max`, `f_min`, `f_max`, `ap_min`, `ap_max`, `VB_max`, `Ra_max`, `kappa_total`, `D`, `L_cut`, `VB_0`, `w1`, and `w2`. The three physics kwargs are discarded without warning.

Probe result:

```
C_wear override honored?  env = 0.00011  requested 0.05
k1 override honored?      env = 12.5     requested 14.0
kappa_total honored?      env = 12.0     requested 12.0
```

Only workpiece geometry varies across the 100 scenarios. `README.md:146` claims robustness "across randomized material/tool configurations", and `docs/LITERATURE_COMPARISON.md:14` claims "high statistical robustness across randomized physical variations". Neither statement describes what the sweep exercises.

**Fix:** add `C_wear`, `k1`, `k2` (and `k3`, `alpha`, `beta`, `gamma_wear`) as overridable kwargs in `MillingDTEnv.__init__`, and have the environment use per-instance copies of the constant dicts rather than the shared module-level objects.

### 4.3 `T_total` is never exported, so the cycle-time metric is dead (P1)

**Files:** `env/milling_env.py:218-230`, `agent/evaluate.py:153`

The `info` dict returned by `step()` contains `pass`, `kappa_rem`, `a_p`, `f_r`, `n`, `Ra`, `MRR`, `VB`, `Energy`, `broken`, and `is_finishing`. There is no `T_total` key, although the environment maintains `self.T_total` internally.

`agent/evaluate.py:153` reads `info.get('T_total', 0)`, which therefore returns 0 on every step. `compute_statistics` at `agent/evaluate.py:43` includes `T_total` in its metric list and reports a confidence interval around it.

Verified against the committed artifact `results/monte_carlo/eval_PPO_MC_v4.csv`:

```
rows: 100
T_total unique: [0] all zero: True
```

**Fix:** add `"T_total": self.T_total` to the `info` dict and re-run the sweep.

### 4.4 The committed ablation artifact is degenerate (P1)

**Files:** `agent/train_rl_ablation.py`, `results/ablation/ablation_results.csv`

The script trains one PPO policy per energy weight and evaluates each over 50 episodes. The committed results table holds three rows with byte-identical metrics and zero variance:

```
w1,w2,MRR_mean,MRR_std,Energy_mean,Energy_std,Time_mean,Time_std,...
1.0,0.0,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
0.5,0.5,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
0.0,1.0,628.31854,0.0,20.352402,0.0,17.142857,0.0,...
```

A deterministic rollout of the three corresponding checkpoints explains why. All three collapse to the same saturated action pair, spindle at 8000 RPM, feed at 0.35 mm/r for the first pass then 0.05 mm/r for the second:

```
w2_0_0: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
w2_0_5: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
w2_1_0: acts=[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
```

The underlying machinery works. The four intermediate checkpoints that are on disk but absent from the CSV do show the expected trade-off, with spindle speed and feed falling as the energy weight rises:

```
w2_0_2: n=6444.5  f_r=0.2457
w2_0_4: n=6289.6  f_r=0.2832
w2_0_6: n=6024.9  f_r=0.2499
w2_0_8: n=6031.1  f_r=0.2435
```

Two further deviations from the plan:

- The plan specifies a Pareto frontier plot of MRR against Energy. `agent/train_rl_ablation.py:139-173` now draws two line plots of Energy and Time against the weight. The committed `results/plots/ablation_pareto_frontier.png` is an orphan from an earlier implementation and is no longer regenerated by any script.
- The plan specifies three seeds and both PPO and SAC. The implementation uses one seed and PPO only.

**Fix:** re-run the ablation across the full weight grid, write the MRR against Energy Pareto frontier, and delete the orphaned PNG.

### 4.5 Version control excludes the core environment (P0, outside the plan)

**Files:** `.gitignore:58`, `.gitignore:77`

The ignore file contains `ENV/` at line 58, intended for virtual environments, and `agent/` at line 77. On a case-insensitive filesystem `ENV/` also matches `env/`.

```
$ git check-ignore -v env/milling_env.py env/__init__.py agent/train_rl_ablation.py agent/run_pipeline.py
.gitignore:58:ENV/	env/milling_env.py
.gitignore:58:ENV/	env/__init__.py
.gitignore:77:agent/	agent/train_rl_ablation.py
.gitignore:77:agent/	agent/run_pipeline.py
```

`git ls-files` lists no path under `env/`, and `git show HEAD:env/milling_env.py` reports the path as existing on disk but not in HEAD. The file that contains every Part 1 fix from the plan has never been committed. The same applies to the ablation script and the pipeline orchestrator.

A fresh clone of this repository cannot construct the environment.

**Fix:** replace `ENV/` with `env/`, `venv/`, and `.venv/`, remove the `agent/` rule, then commit `env/`, `agent/train_rl_ablation.py`, and `agent/run_pipeline.py`.

---

## 5. Unsupported claims

### 5.1 The SAC results in the literature comparison do not exist (P0)

**File:** `docs/LITERATURE_COMPARISON.md`

The document states at line 30: "We implemented and benchmarked both PPO and SAC", and line 22 refers to a `MillingDTEnv-v2` that was never created. The benchmarking table at line 36 carries a full column headed "Our SAC (Eco-Zone)" with four quantitative values, including a 16.4 percent energy reduction and a zero percent failure rate.

A search for SAC across every Python file in the repository returns 0 matches. There is no SAC implementation. `agent/train_rl.py:124` restricts the algorithm choice to PPO, `agent/evaluate.py:89` evaluates PPO only, and `agent/run_experiments.py:131` schedules PPO only. The SAC column is not a result that is hard to reproduce, it is a result that does not exist.

This violates the plan's own success criterion 7 and the "Evidence Over Claims" principle.

**Fix:** either implement SAC end to end (train, evaluate, sweep, benchmark) or remove every SAC claim from the document. The second option is the honest one if there is no time to run the experiments.

### 5.2 README drift (P2)

**File:** `README.md`

| Claim | Actual |
|---|---|
| Line 7: spindle speed 800 to 3200 RPM | `config/physics_constants.py:31-32`: 2000 to 8000 RPM |
| Line 8: feed 0.05 to 0.25 mm/tooth | `config/physics_constants.py:33-34`: 0.05 to 0.35 mm/r |
| Line 50: two hidden layers of 64 units | `agent/train_rl.py:70`: `net_arch=[256, 256]` |
| Lines 75-76: `baselines/genetic_algorithm.py`, `baselines/rule_based.py` | Both are implemented inside `baselines/run_baselines.py` |

The reported experimental findings at `README.md:146-148` do hold against the committed Monte Carlo artifact:

```
Success mean: 1.0
Ra_violation mean: 0.0
VB_violation mean: 0.0
Broken mean: 0.0
```

Subject to the caveat in Section 4.2 that the sweep varies geometry only.

---

## 6. Success criteria status

| # | Criterion | Status | Note |
|---|---|---|---|
| 1 | Passes `gymnasium.check_env()` with no warnings | Met | Verified against `MillingDTEnv` |
| 2 | Two runs with the same seed produce identical results | Met | `tests/test_training.py:57` passes. Bit-level identity of full PPO runs is not separately proven. |
| 3 | Convergence curve monotonic, no divergence or NaN | Met | 16 TensorBoard runs exist under `results/tensorboard/PPO_v4_seed42_1` through `_16` |
| 4 | Monte Carlo sweep reports 95 percent CI for all metrics | Not met | `T_total` is identically zero (Section 4.3), so 1 of the 5 reported metrics is empty |
| 5 | Baselines and environment use identical constants | Met | Verified by reading both consumers of `config/physics_constants.py` |
| 6 | No duplicate plotting code | Met | `viz/plotting_utils.py` is the single source |
| 7 | Ablation study demonstrating the energy objective | Not met | Output is degenerate (Section 4.4); SAC comparison absent (Section 5.1) |

---

## 7. Remediation order

| Order | Action | Effort | Unblocks |
|---|---|---|---|
| 1 | Fix `.gitignore` and commit `env/`, `agent/train_rl_ablation.py`, `agent/run_pipeline.py` | 10 min | Everything else. A reviewer cannot currently read the environment. |
| 2 | Move `import numpy as np` to module scope in `agent/train_rl.py` | 5 min | Parallel multi-seed training (Section 4.1) |
| 3 | Add physics kwargs to `MillingDTEnv.__init__` and export `T_total` in `info` | 30 min | Meaningful Monte Carlo sweep (Sections 4.2, 4.3) |
| 4 | Re-run the ablation across the full weight grid, produce the MRR against Energy Pareto frontier, remove the orphaned PNG | 2 hrs (compute) | Plan items 4.1 and 7 |
| 5 | Remove the SAC claims from `docs/LITERATURE_COMPARISON.md`, or implement SAC | 1 hr, or a full training budget | Honesty of the literature positioning |
| 6 | Correct the README parameter and structure drift | 20 min | Reader accuracy |

Steps 1 and 2 are small and gate everything downstream. Step 3 is required before the Monte Carlo numbers can support the robustness claim they are currently attached to.

---

## Appendix A. Reproducing the findings

```powershell
# Test suite
python -m pytest tests/ -v

# Environment checker
python -c "import sys; sys.path.insert(0,'.'); import gymnasium as gym, env; \
from gymnasium.utils.env_checker import check_env; check_env(gym.make('MillingDTEnv-v1').unwrapped)"

# Section 4.1, queue heartbeat
python -c "import sys; sys.path.insert(0,'.'); import agent.train_rl as t; \
q=[]; F=type('F',(),{'put_nowait':lambda s,x: q.append(x)}); \
cb=t.ProgressQueueCallback(F()); cb.init_callback(None); \
cb.locals={'infos':[{'reward':1.0}]}; cb._on_step(); print(q)"

# Section 4.2, ignored physics kwargs
python -c "import sys; sys.path.insert(0,'.'); import gymnasium as gym, env; \
u=gym.make('MillingDTEnv-v1', C_wear=0.05, k1=14.0).unwrapped; \
print(u.tool_wear_cfg['C_wear'], u.roughness_cfg['k1'])"

# Section 4.5, version control coverage
git check-ignore -v env/milling_env.py agent/train_rl_ablation.py
git ls-files | Select-String 'env/'
```

## Appendix B. Artifacts on disk at audit time

| Path | Contents |
|---|---|
| `models/PPO_seed42/` | Primary policy plus `vecnormalize.pkl` |
| `models/ablation_w2_0_0/` through `w2_1_0/` | 8 ablation policies (weights 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0) |
| `results/monte_carlo/eval_PPO_MC_v4.csv` | 100 scenario rows. Success 1.0, violations 0.0, `T_total` all zero |
| `results/ablation/ablation_results.csv` | 3 rows, degenerate (Section 4.4) |
| `results/plots/ablation_pareto_frontier.png` | Orphan, not regenerated by any current script |
| `results/tensorboard/PPO_v4_seed42_1` to `_16` | 16 training runs |
