# Digital Twin CNC Milling - Claude Code Project Guide

## Project Commands
- Run tests: `pytest tests/ -v`
- Run PPO Training: `python agent/train_rl.py --steps 200000 --seed 42`
- Run Monte Carlo Evaluation: `python agent/evaluate.py`
- Run Baselines Benchmark: `python viz/plot_baselines_comparison.py`
- Launch Interactive Digital Twin Simulator: `streamlit run app.py`

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
