"""
agent/run_pipeline.py

Master orchestrator script.
Executes the full pipeline sequentially:
1. Primary PPO Training
2. Conditioned Policy Training
3. Monte Carlo Evaluation (Primary Model)
4. Conditioned Policy Weight Sweep
5. Ablation Study (Trains Pareto Frontier Models)
6. Classical Baselines
7. Viz Generation (Training Curves, MC Distributions, Baseline Comparison)
8. Result Tables (rebuilds results/RESULTS.md)
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
    parser.add_argument("--steps", type=int, default=1_000_000, help="Timesteps for primary training")
    parser.add_argument("--ablation_steps", type=int, default=100_000, help="Timesteps per ablation model")
    parser.add_argument("--scenarios", type=int, default=50, help="Scenarios per ablation policy")
    parser.add_argument("--slots", type=int, default=7, help="Parallel training slots for the ablation")
    args = parser.parse_args()

    print("="*80)
    print("  DIGITAL TWIN CNC MILLING - MASTER PIPELINE")
    print("="*80)

    # 1. Primary Model Training
    run_cmd(f"python agent/train_rl.py --algo PPO --steps {args.steps} --seed 42 --randomize")

    # 2. Conditioned Policy Training
    run_cmd(f"python agent/train_rl_conditioned.py --steps {args.steps} --seed 42")

    # 3. Monte Carlo Evaluation
    run_cmd("python agent/evaluate.py")

    # 4. Conditioned policy weight sweep
    run_cmd("python agent/evaluate_conditioned.py --scenarios 100 --compare")

    # 5. Ablation Study Training (Trains Pareto Frontier)
    run_cmd(f"python agent/train_rl_ablation.py --steps {args.ablation_steps} "
            f"--scenarios {args.scenarios} --slots {args.slots}")

    # 6. Classical Baselines
    run_cmd("python baselines/run_baselines.py")

    # 7. Generate all Visualizations
    run_cmd("python viz/plot_results.py")
    run_cmd("python viz/plot_eval.py")
    run_cmd("python viz/plot_baselines_comparison.py")

    # 8. Rebuild the result tables quoted in the documentation
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
