# Digital Twin CNC Milling - Claude Code Project Guide

## Project Commands
- Run tests: `pytest tests/ -v`
- Train the primary policy: `python agent/train_rl.py --steps 1000000 --seed 42 --randomize`
- Train the conditioned policy: `python agent/train_rl_conditioned.py --steps 1000000 --seed 42`
- Run the Monte Carlo sweep: `python agent/evaluate.py`
- Run the energy weight ablation: `python agent/train_rl_ablation.py --steps 150000 --scenarios 50 --slots 7`
- Run the classical baselines: `python baselines/run_baselines.py`
- Rebuild the result tables: `python agent/summarize_results.py`
- Launch the interactive Digital Twin simulator: `streamlit run app.py`

## Documentation
- `README.md`: overview, method, and the measured findings
- `PROJECT.md`: every plan and reference document, in one file
  - Part 1: the environment contract and the acceptance status
  - Part 2: the conditioned policy, the transfer experiment, the controller
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
