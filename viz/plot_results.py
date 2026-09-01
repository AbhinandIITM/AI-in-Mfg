import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs():
    log_dir = r"G:\My Drive\IITM\AI in Mfg\logs\tensorboard"
    plot_path = r"G:\My Drive\IITM\AI in Mfg\plots\training_curves.png"

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    runs = glob.glob(os.path.join(log_dir, "*"))

    plt.figure(figsize=(10, 6))
    colors = {"PPO": "blue", "SAC": "red"}

    for run_path in runs:
        if not os.path.isdir(run_path): continue
        run_name = os.path.basename(run_path)
        
        algo = "PPO" if "PPO" in run_name else "SAC"
        
        event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
        if not event_files: continue
        
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        
        try:
            scalars = ea.Scalars('rollout/ep_rew_mean')
            steps = [s.step for s in scalars]
            rewards = [s.value for s in scalars]
            
            plt.plot(steps, rewards, label=run_name, color=colors[algo], alpha=0.7, linewidth=2)
        except KeyError:
            print(f"Key rollout/ep_rew_mean not found in {run_name}")

    plt.title("Training Curves: PPO vs SAC (v3 Improvements)")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Normalized Episode Reward Mean")
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_tensorboard_logs()
