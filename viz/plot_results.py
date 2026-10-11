import os
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from viz.plotting_utils import set_style

def plot_tensorboard_logs():
    log_dir = PROJECT_ROOT / "results" / "tensorboard"
    plot_path = PROJECT_ROOT / "results" / "plots" / "training_curves.png"

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    runs = [r for r in glob.glob(os.path.join(str(log_dir), "*")) if "PPO" in os.path.basename(r)]

    set_style()
    plt.figure(figsize=(10, 6))

    for run_path in runs:
        if not os.path.isdir(run_path): continue
        run_name = os.path.basename(run_path)

        event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
        if not event_files: continue

        ea = EventAccumulator(event_files[0])
        ea.Reload()

        try:
            scalars = ea.Scalars('rollout/ep_rew_mean')
            steps = [s.step for s in scalars]
            rewards = [s.value for s in scalars]

            plt.plot(steps, rewards, label=run_name, color="blue", alpha=0.7, linewidth=2)
        except KeyError:
            print(f"Key rollout/ep_rew_mean not found in {run_name}")

    plt.title("Training Curves: PPO")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Normalized Episode Reward Mean")


    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_tensorboard_logs()
