import os
import re
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from viz.plotting_utils import set_style

def latest_runs_per_name(run_dirs):
    """Keep the newest run of each name.

    Stable-Baselines3 appends a counter to the TensorBoard run name, so a
    name that was trained more than once leaves several directories behind.
    Only the newest one describes the model that is actually in models/.
    """
    newest = {}
    for run_dir in run_dirs:
        base = os.path.basename(run_dir)
        match = re.match(r"^(.*)_(\d+)$", base)
        if match:
            name, index = match.group(1), int(match.group(2))
        else:
            name, index = base, 0
        if name not in newest or index > newest[name][0]:
            newest[name] = (index, run_dir)
    return {name: path for name, (_, path) in newest.items()}

def plot_tensorboard_logs():
    log_dir = PROJECT_ROOT / "results" / "tensorboard"
    plot_path = PROJECT_ROOT / "results" / "plots" / "training_curves.png"

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    run_dirs = latest_runs_per_name(glob.glob(os.path.join(str(log_dir), "*")))

    set_style()
    plt.figure(figsize=(10, 6))

    colors = {"PPO_primary_seed42": "#1f77b4",
              "PPO_conditioned_seed42": "#0d9488",
              "PPO_conditioned_150k_seed42": "#94a3b8"}

    plotted = 0
    for name, run_path in sorted(run_dirs.items()):
        if not name.startswith("PPO"):
            continue
        if not os.path.isdir(run_path):
            continue

        event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
        if not event_files:
            continue

        ea = EventAccumulator(event_files[0])
        ea.Reload()

        try:
            scalars = ea.Scalars('rollout/ep_rew_mean')
        except KeyError:
            print(f"Key rollout/ep_rew_mean not found in {name}")
            continue

        steps = [s.step for s in scalars]
        rewards = [s.value for s in scalars]
        label = name.replace("PPO_primary_seed42", "PPO (primary)")
        label = label.replace("PPO_conditioned_150k_seed42", "PPO (conditioned, 150k steps)")
        label = label.replace("PPO_conditioned_seed42", "PPO (conditioned, 1M steps)")
        plt.plot(steps, rewards, label=label, color=colors.get(name, "#64748b"),
                 alpha=0.9, linewidth=2)
        plotted += 1

    plt.title("Training Curves: Mean Episode Reward")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Normalized Episode Reward Mean")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path} ({plotted} runs)")

if __name__ == "__main__":
    plot_tensorboard_logs()
