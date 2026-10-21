# Digital Twin CNC Milling - Claude Code Project Guide

## Repository Layout
- `MillingDT/` is the library package: the environment, the physical model, the classical baselines, the controller and the seeding.
  - `MillingDT/config/` holds the configuration, in YAML: `physics_constants.yaml` (the calibrated constants, the reward weights, the scenario tables) and `paths.yaml` (where the project keeps its runs, logs, plots and tables). `MillingDT/config/__init__.py` loads both and exposes the constants and the paths, so no other module parses YAML.
  - `MillingDT/env/milling_env.py` is the environment, `MillingDT/config/physics.py` the physical model, `MillingDT/config/scenarios.py` the randomized scenarios.
  - `MillingDT/baselines/receding_horizon.py` is the controller, `MillingDT/agent/controller_init.py` the initialization of the policy from that controller.
- `experiments/` holds the entry points: `train_rl.py`, `train_rl_ablation.py`, `evaluate.py`, `run_baselines.py`, `transfer_experiment.py`, `controller_experiment.py`, `run_experiments.py`, `run_pipeline.py` and `summarize_results.py`. Run them as modules, for example `python -m experiments.train_rl`.
- `viz/` holds the figures: `plot_results.py`, `plot_eval.py`, `plot_baselines_comparison.py`, `plot_controller_comparison.py` and the shared `plotting_utils.py`. Run them the same way, for example `python -m viz.plot_results`. Every baseline that carries a row in a comparison table needs a label in the figure script as well, otherwise its bar is dropped without a word.
- `results/` holds every run, one directory per training configuration, named `run_<config>`. Each run writes to `results/run_<config>/models/` (the policy and its normalization statistics), `results/run_<config>/log/` (the training curve, `progress.csv`), `results/run_<config>/metrics/` (the evaluation tables of that policy) and `results/run_<config>/plots/` (the figures of that policy).
- `results/evaluation/` holds the tables that compare several runs, `results/plots/` the cross-run figures, and `results/RESULTS.md` the generated summary of every artifact.
- The run names are fixed by `MillingDT/config/paths.yaml`: `run_PPO_seed42` is the production policy, `run_ablation_w2_<w2>_seed<seed>` the 21 ablation policies, `run_PPO_controller_init_seed42`, `run_transfer_source_seed42` and `run_transfer_scratch_seed42` the transfer arms.

## Project Commands
- Run tests: `pytest tests/ -v`
- Train the production policy: `python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight`
- Run the Monte Carlo sweep: `python -m experiments.evaluate --scenarios 100`
- Run the runtime weight sweep: `python -m experiments.evaluate --sweep-weight --scenarios 100`
- Compare the one policy against the per-weight policies: `python -m experiments.evaluate --compare-only`
- Run the energy weight ablation: `python -m experiments.train_rl_ablation --steps 150000 --scenarios 50 --slots 7`
- Run the classical baselines: `python -m experiments.run_baselines`
- Run the transfer experiment: `python -m experiments.transfer_experiment 300000 50000`
- Run the controller initialization experiment: `python -m experiments.controller_experiment 1000000 2000`
- Rebuild the result tables: `python -m experiments.summarize_results`
- Build the figures: `python -m viz.plot_results`, `python -m viz.plot_eval`, `python -m viz.plot_baselines_comparison`, `python -m viz.plot_controller_comparison`
- Run the full pipeline: `python -m experiments.run_pipeline --steps 1000000 --ablation_steps 150000`
- Launch the interactive Digital Twin simulator: `streamlit run app.py`

## Documentation
- `README.md`: overview, method, and the measured findings
- `PROJECT.md`: every plan and reference document, in one file
  - Part 1: the environment contract and the acceptance status
  - Part 2: the method upgrade, the one policy, the transfer experiment, the controller, the controller initialization
  - Part 3: position against the three papers in `docs/papers/`
  - Part 4: the design of the Streamlit workstation
  - Part 5: where every physics equation comes from
  - Parts 6 and 7: the original improvement plan and the audit of it
- `results/RESULTS.md`: the result tables, generated from the artifacts

<!-- antislop:start -->
## antislop
For UI, copy, people, mobile layout, or code comments work, read `.agents/skills/antislop/SKILL.md` (core) and then the skill for the task:
- UI / visual: `.agents/skills/antislop-ui/SKILL.md`
- Copy & text: `.agents/skills/antislop-copywriting/SKILL.md`
- People: `.agents/skills/antislop-human/SKILL.md`
- Mobile / responsive: `.agents/skills/antislop-layoutmobile/SKILL.md`
- Code comments: `.agents/skills/antislop-code/SKILL.md`
Before starting, follow the core's "Two Usage Modes" section in strict order: explicit session instruction first, then global preference, then ask. A session instruction always wins. For a resolved mode, say `antislop active: <mode> (session override).` or `antislop active: <mode> (global preference).` once before presenting findings or making edits, using the actual mode and source.
<!-- antislop:end -->
