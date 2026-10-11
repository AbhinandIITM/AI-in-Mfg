# Digital Twin CNC Milling - AGENTS Configuration

This repository hosts an intelligent manufacturing Digital Twin for dynamic CNC milling parameter optimization using Deep Reinforcement Learning (PPO).

## Guidelines & Principles
- All UI work adheres to anti-slop guidelines: no generic purple glows, no AI-slop marketing copy, authentic engineering telemetry, and explicit domain-grounded manufacturing visuals.
- Reproducibility and physical consistency are enforced via `config/physics_constants.py` and `utils/seeding.py`.
- Unit tests (`pytest tests/ -v`) must remain 100% green with 0 warnings.

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
