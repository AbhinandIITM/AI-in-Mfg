# Agent Skills Guide for Claude Code & Antigravity (AGY)

This project integrates [Addy Osmani's **agent-skills**](https://github.com/addyosmani/agent-skills), encoding production-grade software engineering workflows, quality gates, and best practices directly into your AI coding assistants.

Both **Claude Code** and **Antigravity (`agy`)** are fully configured at the **global environment level** and locally inside this workspace.

---

## Quick Navigation

- [Installation & Setup Status](#installation--setup-status)
- [The 6-Phase Engineering Lifecycle](#the-6-phase-engineering-lifecycle)
- [Commands Reference & Usage](#commands-reference--usage)
  - [`/spec` (Spec-Driven Development)](#1-spec---define-what-to-build)
  - [`/plan` (Atomic Task Breakdown)](#2-plan---plan-how-to-build-it)
  - [`/build` and `/build auto` (Incremental Implementation)](#3-build-and-build-auto---build-incrementally)
  - [`/test` (Test-Driven Development)](#4-test---prove-it-works)
  - [`/constraints` (Architecture & Invariant Enforcement)](#5-constraints---set-the-quality-bar)
  - [`/review` (5-Axis Quality Review)](#6-review---review-before-merge)
  - [`/webperf` (Performance & Benchmarking)](#9-webperf---measure-before-optimizing)
  - [`/train-rl` & `/hf-train` (RL & ML Training)](#10-train-rl-and-hf-train---reinforcement-learning--ml-training)
- [Specialized RL, PyTorch & Agent Training Skills Guide](RL_AND_AGENT_TRAINING_SKILLS_GUIDE.md)
- [How to Invoke in Claude Code vs Antigravity](#how-to-invoke-in-claude-code-vs-antigravity)
- [Complete Catalog of Skills](#complete-catalog-of-skills)
- [Workflow Examples for This Repository (AI in Manufacturing)](#workflow-examples-for-this-repository)

---

## Installation & Setup Status

Both agents are fully provisioned:

| Component | Claude Code | Antigravity CLI (`agy`) |
| :--- | :--- | :--- |
| **Global Plugin** | Installed via marketplace (`agent-skills@addy-agent-skills`) | Installed via native plugin (`~/.gemini/config/plugins/agent-skills`) |
| **Workspace Config** | Local commands in `.claude/commands/` and rules in `.claude/rules/` | Local skills in `.agents/skills/` and `.agents/references/` |
| **Discovery** | Auto-loaded on session start in this folder | Auto-discovered hierarchically by `agy` |

---

## The 6-Phase Engineering Lifecycle

```
  DEFINE          PLAN           BUILD          VERIFY         REVIEW          SHIP
                               
  Idea    Spec    Code    Test     QA      Go  
 Refine        PRD        Impl       Debug        Gate        Live 
                               
  /spec          /plan          /build        /test         /review       /ship
```

### Core Philosophy
1. **Never code without a spec**: Write clear interfaces, edge cases, and acceptance criteria first.
2. **Atomic tasks**: Break work into independent, testable units that can be verified in isolation.
3. **Red-Green-Refactor**: Write failing tests first; implement the minimum code to pass; refactor with confidence.
4. **Autonomous execution with human gates**: Use `/build auto` when you want the agent to implement an entire plan end-to-end, pausing only for risky actions or test failures.

---

## Commands Reference & Usage

### 1. `/spec` - Define What to Build
- **Key Principle**: *Spec before code.*
- **What it does**: Interrogates requirements, clarifies ambiguity, documents assumptions, and drafts a comprehensive engineering specification with data models, error handling, and acceptance criteria before creating implementation code.
- **Claude Code**: `/spec [feature description]`
- **Antigravity CLI**: `/agent-skills:spec-driven-development [feature description]`
- **Example**:
  ```text
  /spec Add priority dispatch heuristic baseline to compare against PPO agent
  ```

---

### 2. `/plan` - Plan How to Build It
- **Key Principle**: *Small, atomic tasks.*
- **What it does**: Takes an approved specification or requirement and translates it into an ordered task dependency graph. Each task includes:
  - Exact files modified or created.
  - Test criteria for verifying completion.
  - Rollback strategies and risks.
- **Claude Code**: `/plan [spec or task objective]`
- **Antigravity CLI**: `/agent-skills:planning-and-task-breakdown [objective]`
- **Example**:
  ```text
  /plan Break down the integration of real-time machine breakdown simulation in env/mfg_env.py
  ```

---

### 3. `/build` and `/build auto` - Build Incrementally
- **Key Principle**: *One vertical slice at a time.*
- **What it does**:
  - `/build`: Takes the next pending task from the plan, implements only that piece, runs verification, and halts for review.
  - `/build auto`: Executes all planned tasks sequentially without requiring manual prompts between tasks. It commits at each verified step and pauses only if a test fails or an ambiguous choice arises.
- **Claude Code**: `/build` or `/build auto`
- **Antigravity CLI**: `/agent-skills:incremental-implementation`
- **Example**:
  ```text
  /build auto Implement the planned observation space normalizer and action masking wrappers
  ```

---

### 4. `/test` - Prove It Works
- **Key Principle**: *Tests are proof.*
- **What it does**: Enforces strict Test-Driven Development (TDD). It generates edge-case tests, asserts expected invariants, verifies failure reproduction (Red), implements the solution (Green), and checks code coverage.
- **Claude Code**: `/test [module or behavior under test]`
- **Antigravity CLI**: `/agent-skills:test-driven-development [module or behavior]`
- **Example**:
  ```text
  /test tests/test_env.py verify gym environment step resets and reward calculation bounds
  ```

---

### 5. `/constraints` - Set the Quality Bar
- **Key Principle**: *Decide it once, enforce it everywhere.*
- **What it does**: Establishes or reviews architectural invariants, security rules, performance budgets, and style guidelines that every subsequent agent prompt must respect.
- **Claude Code**: `/constraints`
- **Antigravity CLI**: `/agent-skills:constraint-driven-development`
- **Example**:
  ```text
  /constraints Require vectorized environment compatibility with Stable-Baselines3 and strict float32 observation arrays
  ```

---

### 6. `/review` - Review Before Merge
- **Key Principle**: *Improve code health.*
- **What it does**: Runs a comprehensive 5-axis review:
  1. **Correctness**: Logic bugs, edge conditions, race conditions.
  2. **Security**: Inputs, sanitization, unsafe deserialization.
  3. **Performance**: Computational bottlenecks, memory footprint, vectorization.
  4. **Maintainability**: Modularity, readable variable naming, typing.
  5. **Test coverage**: Uncovered regression scenarios.
- **Claude Code**: `/review`
- **Antigravity CLI**: `/agent-skills:code-review-and-quality`
- **Example**:
  ```text
  /review Check recent modifications in agent/train.py and utils/metrics.py
  ```

---

### 7. `/code-simplify` - Clean and Simplify
- **Key Principle**: *Clarity over cleverness.*
- **What it does**: Eliminates unnecessary abstractions, dead branches, nested conditionals, and over-engineered logic without altering external contracts or behavior.
- **Claude Code**: `/code-simplify [file/directory]`
- **Antigravity CLI**: `/agent-skills:code-simplification [file/directory]`
- **Example**:
  ```text
  /code-simplify viz/live_dashboard.py
  ```

---

### 8. `/ship` - Ship to Production
- **Key Principle**: *Faster is safer when verified.*
- **What it does**: Runs a pre-flight launch verification:
  - Executes full test suite.
  - Validates documentation, dependencies, and environment reproducibility.
  - Generates clear changelogs and release notes.
- **Claude Code**: `/ship`
- **Antigravity CLI**: `/agent-skills:shipping-and-launch`
- **Example**:
  ```text
  /ship Prepare final release for Manufacturing Optimization Evaluation v1.0
  ```

---

### 9. `/webperf` - Measure Before Optimizing
- **Key Principle**: *Measure, profile, optimize.*
- **What it does**: Audits performance (runtime benchmarks, web dashboards, rendering, memory overhead) and recommends targeted optimizations based on profiling data.
- **Claude Code**: `/webperf`
- **Antigravity CLI**: `/agent-skills:performance-optimization`
- **Example**:
  ```text
  /webperf Profile environment step throughput and vector rollout memory usage
  ```

---

### 10. `/train-rl` and `/hf-train` - Reinforcement Learning & ML Training
- **Key Principle**: *Modern Gymnasium standards, stable gradients, and verified recipes.*
- **What it does**:
  - `/train-rl`: Audits Gym environment step loops (5-value unpacks), wires PyTorch/TorchRL algorithms (ClipPPOLoss, GAE, PPO, SAC), applies reward shaping, and conducts rollout verification.
  - `/hf-train`: Orchestrates Hugging Face LLM training jobs, TRL alignment (PPO, GRPO, DPO), dataset preparation, and Trackio telemetry.
- **Claude Code**: `/train-rl` or `/hf-train`
- **Antigravity CLI**: `/mastering-pytorch-rl-nlp`, `/art-train-rl`, `/atraining-rl-recipe`, or `/huggingface-llm-trainer`
- **Example**:
  ```text
  /train-rl Verify env/milling_dt_env.py step outputs, reward scaling, and vectorized rollout performance
  ```

> For deep documentation on all 5 added suites, see [**RL_AND_AGENT_TRAINING_SKILLS_GUIDE.md**](RL_AND_AGENT_TRAINING_SKILLS_GUIDE.md).

---

## How to Invoke in Claude Code vs Antigravity

Because Claude Code and Antigravity handle slash commands and skill naming differently, refer to this mapping:


| Workflow Intent | Claude Code Command | Antigravity (`agy`) Invocation |
| :--- | :--- | :--- |
| **Draft Spec** | `/spec` | `/agent-skills:spec-driven-development` |
| **Create Plan** | `/plan` | `/agent-skills:planning-and-task-breakdown` |
| **Execute Build** | `/build` or `/build auto` | `/agent-skills:incremental-implementation` |
| **Write / Run Tests** | `/test` | `/agent-skills:test-driven-development` |
| **Review Code** | `/review` | `/agent-skills:code-review-and-quality` |
| **Enforce Invariants** | `/constraints` | `/agent-skills:constraint-driven-development` |
| **Simplify Code** | `/code-simplify` | `/agent-skills:code-simplification` |
| **Release / Ship** | `/ship` | `/agent-skills:shipping-and-launch` |
| **Profile & Optimize**| `/webperf` | `/agent-skills:performance-optimization` |
| **Requirements Interrogation** | `/interview-me` | `/agent-skills:interview-me` |
| **RL & Gym Workflows** | `/train-rl` | `/mastering-pytorch-rl-nlp` or `/art-train-rl` |
| **Hugging Face / LLM Training** | `/hf-train` | `/huggingface-llm-trainer` or `/trl-training` |


> **Pro-tip for Antigravity (`agy`)**:
> You can also trigger any skill using **natural language prompts** without slash commands. `agy` uses progressive disclosure and automatically loads the skill when relevant:
> - *"Use spec-driven development to design our new reward function."*
> - *"Perform a 5-axis code quality review on the environment step method."*
> - *"Follow test-driven development to add unit tests for the baseline scheduler."*

---

## Complete Catalog of 25 Skills

All 25 skills are available in this project under `.agents/skills/`:

1. `spec-driven-development` - Spec creation before implementation.
2. `planning-and-task-breakdown` - Atomic task graphs and dependency ordering.
3. `incremental-implementation` - Step-by-step verified feature delivery.
4. `test-driven-development` - Strict red-green-refactor loop.
5. `code-review-and-quality` - 5-axis code reviews and health checks.
6. `code-simplification` - Refactoring for readability and minimalism.
7. `constraint-driven-development` - System boundaries, SLAs, and architecture guardrails.
8. `debugging-and-error-recovery` - Hypothesis-driven bug reproduction and root-cause analysis.
9. `performance-optimization` - Profiling, bottleneck analysis, and caching.
10. `interview-me` - Interactive Q&A to extract underspecified requirements.
11. `idea-refine` - Brainstorming, user story mapping, and concept shaping.
12. `source-driven-development` - Grounding implementations in authoritative source documents.
13. `doubt-driven-development` - Adversarial thinking, edge-case probing, stress testing.
14. `api-and-interface-design` - Clean, composable, developer-friendly interfaces.
15. `frontend-ui-engineering` - Resilient, component-driven UI and dashboards.
16. `browser-testing-with-devtools` - Automated browser audits and headless testing.
17. `observability-and-instrumentation` - Metrics, logging, traces, and alert design.
18. `security-and-hardening` - Vulnerability mitigation, input validation, secret hygiene.
19. `git-workflow-and-versioning` - Clean commit history, branch strategies, semver.
20. `ci-cd-and-automation` - Fast, reliable pipelines and checks.
21. `documentation-and-adrs` - Architecture Decision Records (ADRs) and living documentation.
22. `deprecation-and-migration` - Backward compatibility and zero-downtime migrations.
23. `shipping-and-launch` - Verification checklists, rollouts, and runbooks.
24. `context-engineering` - Token optimization and prompt context structuring for AI models.
25. `using-agent-skills` - Meta-skill guiding agents on how to combine multiple skills effectively.

---

## Workflow Examples for This Repository

Here are hands-on scenarios customized for the **AI in Manufacturing** project:

### Scenario 1: Developing a New Reward Shaping Function
```bash
# In Claude Code:
/spec Formulate a potential-based reward shaping mechanism in env/ to encourage higher machine throughput while penalizing queue backlog
/plan
/build auto
/test tests/test_env.py -k test_reward_shaping
/review
```

### Scenario 2: Debugging High Variance or NaN Gradients in RL Training
```bash
# In Claude Code:
/test tests/test_agent.py Reproduce NaN action distribution logits under high backlog states

# In Antigravity CLI:
agy "Use debugging-and-error-recovery to isolate why PPO loss explodes after step 5000 in agent/train.py"
```

### Scenario 3: Pre-submission Code Quality and Evaluation Audit
```bash
# In Claude Code:
/constraints Verify that models/, plots/, and video recordings are excluded from git and follow reproducibility guidelines
/review
/ship
```
