"""
experiments/run_pipeline.py

Master orchestrator script.
Executes the full pipeline sequentially:
1. Production PPO Training (one policy over the whole energy weight range)
2. Monte Carlo Evaluation (Production Model)
3. Runtime Weight Sweep (the production policy asked for every weight)
4. Ablation Study (Trains Pareto Frontier Models)
5. Classical Baselines
6. Transfer to a harder material
7. Controller Initialization Experiment
8. Robustness Experiment (model error and a noisy wear measurement)
9. Batch Experiment (the tool life shared across the parts)
10. Viz Generation (Training Curves, MC Distributions, Baseline Comparison)
11. Result Tables (rebuilds results/RESULTS.md)

Design note: this script drives the *pipeline stages* through subprocess calls
(python -m experiments.train_rl, python -m experiments.run_baselines, etc.).
The parallel training of the ablation models is delegated to
experiments/run_experiments.py, which uses multiprocessing to supervise a pool
of training workers with heartbeat monitoring. The two strategies exist because
run_pipeline.py sequences dependent stages (each stage needs the previous
stage's artifacts), while run_experiments.py parallelizes *independent* jobs
within one stage (the ablation grid). Merging them would conflate sequential
orchestration with parallel worker supervision.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_cmd(cmd):
    print(f"\n{'-'*80}")
    print(f"[>>>] EXECUTING: {cmd}")
    print(f"{'-'*80}\n")
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)
    print(f"\n[OK] COMPLETED: {cmd}\n")

def main():
    parser = argparse.ArgumentParser(description="Run full CNC Digital Twin RL Pipeline")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Timesteps for the production policy")
    parser.add_argument("--ablation_steps", type=int, default=150_000,
                        help="Timesteps per ablation model")
    parser.add_argument("--eval_scenarios", type=int, default=100,
                        help="Scenarios per policy in the Monte Carlo and weight sweeps")
    parser.add_argument("--scenarios", type=int, default=50,
                        help="Scenarios per ablation policy")
    parser.add_argument("--slots", type=int, default=7,
                        help="Parallel training slots for the ablation")
    parser.add_argument("--transfer_steps", type=int, default=300_000,
                        help="Steps for the source policy of the transfer experiment")
    parser.add_argument("--adapt_steps", type=int, default=50_000,
                        help="Adaptation budget on the target material")
    parser.add_argument("--controller_episodes", type=int, default=2000,
                        help="Controller episodes recorded for the initialization")
    args = parser.parse_args()

    print("="*80)
    print("  DIGITAL TWIN CNC MILLING - MASTER PIPELINE")
    print("="*80)

    # 1. The production policy: the weight and the scenario are drawn per
    # episode, so one network covers the whole energy weight range.
    run_cmd(f"python -m experiments.train_rl --steps {args.steps} --seed 42 "
            f"--randomize --randomize-weight")

    # 2. Monte Carlo evaluation of that policy
    run_cmd(f"python -m experiments.evaluate --scenarios {args.eval_scenarios}")

    # 3. The same policy asked for every point of the trade-off. The weight is
    # a query, not a retrain.
    run_cmd(f"python -m experiments.evaluate --sweep-weight --scenarios {args.eval_scenarios}")

    # 4. Ablation Study Training (Trains Pareto Frontier)
    run_cmd(f"python -m experiments.train_rl_ablation --steps {args.ablation_steps} "
            f"--scenarios {args.scenarios} --slots {args.slots}")

    # 5. Classical Baselines
    run_cmd("python -m experiments.run_baselines")

    # 6. Transfer to a harder material
    run_cmd(f"python -m experiments.transfer_experiment {args.transfer_steps} "
            f"{args.adapt_steps}")

    # 7. The controller initialization experiment. It trains its own policy,
    # so it runs before the figures, which read its table.
    run_cmd(f"python -m experiments.controller_experiment {args.steps} "
            f"{args.controller_episodes}")

    # 8. The robustness experiment: the same two arms on a machine the
    # controller has the wrong model of, and with a noisy wear measurement.
    run_cmd("python -m experiments.robustness_experiment")

    # 9. The batch experiment: the tool life is shared across the parts.
    run_cmd(f"python -m experiments.train_rl --env-id MillingDTBatchEnv-v1 "
            f"--steps {args.steps} --seed 42 --randomize "
            f"--run-name PPO_batch_seed42")
    run_cmd("python -m experiments.batch_experiment")

    # 10. Generate all Visualizations
    run_cmd("python -m viz.plot_results")
    run_cmd("python -m viz.plot_eval")
    run_cmd("python -m viz.plot_baselines_comparison")
    run_cmd("python -m viz.plot_controller_comparison")
    run_cmd("python -m viz.plot_robustness")
    run_cmd("python -m viz.plot_batch_comparison")
    run_cmd("python -m viz.plot_reward_sensitivity")

    # 11. Rebuild the result tables quoted in the documentation
    run_cmd("python -m experiments.summarize_results")

    print("="*80)
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(" Next Steps:")
    print("   - Run the digital twin: streamlit run app.py")
    print("   - View the runs:        results/run_<name>/")
    print("   - View the figures:     results/plots/")
    print("   - View the tables:      results/RESULTS.md")
    print("="*80)

if __name__ == "__main__":
    main()
