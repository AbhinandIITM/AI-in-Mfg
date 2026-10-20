"""
agent/run_pipeline.py

Master orchestrator script.
Executes the full pipeline sequentially:
1. Production PPO Training (one policy over the whole energy weight range)
2. Monte Carlo Evaluation (Production Model)
3. Runtime Weight Sweep (the production policy asked for every weight)
4. Ablation Study (Trains Pareto Frontier Models)
5. Classical Baselines
6. Viz Generation (Training Curves, MC Distributions, Baseline Comparison)
7. Result Tables (rebuilds results/RESULTS.md)

Design note: this script drives the *pipeline stages* through subprocess calls
(python agent/train_rl.py, python baselines/run_baselines.py, etc.).
The parallel training of the ablation models is delegated to
agent/run_experiments.py, which uses multiprocessing to supervise a pool of
training workers with heartbeat monitoring. The two strategies exist because
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
    args = parser.parse_args()

    print("="*80)
    print("  DIGITAL TWIN CNC MILLING - MASTER PIPELINE")
    print("="*80)

    # 1. The production policy: the weight and the scenario are drawn per
    # episode, so one network covers the whole energy weight range.
    run_cmd(f"python agent/train_rl.py --steps {args.steps} --seed 42 "
            f"--randomize --randomize-weight")

    # 2. Monte Carlo evaluation of that policy
    run_cmd(f"python agent/evaluate.py --scenarios {args.eval_scenarios}")

    # 3. The same policy asked for every point of the trade-off. The weight is
    # a query, not a retrain.
    run_cmd(f"python agent/evaluate.py --sweep-weight --scenarios {args.eval_scenarios}")

    # 4. Ablation Study Training (Trains Pareto Frontier)
    run_cmd(f"python agent/train_rl_ablation.py --steps {args.ablation_steps} "
            f"--scenarios {args.scenarios} --slots {args.slots}")

    # 5. Classical Baselines
    run_cmd("python baselines/run_baselines.py")

    # 6. Generate all Visualizations
    run_cmd("python viz/plot_results.py")
    run_cmd("python viz/plot_eval.py")
    run_cmd("python viz/plot_baselines_comparison.py")

    # 7. Rebuild the result tables quoted in the documentation
    run_cmd("python agent/summarize_results.py")

    print("="*80)
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(" Next Steps:")
    print("   - Run the digital twin: streamlit run app.py")
    print("   - View plots in:        results/plots/")
    print("   - View logs in:         results/")
    print("="*80)

if __name__ == "__main__":
    main()
