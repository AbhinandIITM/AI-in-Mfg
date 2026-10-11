# Modern Parallel Progress Tracking (Outgrowing Disk I/O)

In the past, we relied on the **Slot-Worker Pattern** where child processes wrote their progress to temporary text files (`progress_seedX.txt`) and a parent process polled those files. 

While that was resilient, we have **outgrown** it for several reasons:
1. **Disk I/O Bottlenecks**: High-frequency writes stall RL loops.
2. **Zombie Files**: Crashes leave leftover `.txt` files.
3. **Subprocess Overhead**: `subprocess.Popen` separates memory completely, making metrics sharing difficult.

## The Modern Standard: `multiprocessing.Queue` (IPC)

Instead of using `subprocess.Popen` and writing to disk, we now import the training function directly and run it via `multiprocessing.Process`. Progress is communicated instantly and safely through RAM using an IPC Queue.

### 1. The Worker Script (`train_rl.py`)
Modify your training function to accept an optional `queue`.

```python
import time
from tqdm import tqdm

def train_agent(total_steps, seed, queue=None):
    # If running standalone, use a normal tqdm bar
    pbar = tqdm(total=total_steps, desc=f"Seed {seed}") if queue is None else None

    for step in range(1, total_steps + 1):
        # ... Do work here ...
        time.sleep(0.01) # Simulated work
        
        loss = 0.5 # Example metric
        
        if queue:
            # Send progress & metrics directly to parent via RAM (no disk I/O!)
            if step % 100 == 0:
                queue.put({"seed": seed, "step": step, "loss": loss})
        else:
            pbar.update(1)
            pbar.set_postfix({"loss": loss})
            
    if queue:
        queue.put({"seed": seed, "done": True})
    else:
        pbar.close()
```

### 2. The Manager Script (`run_experiments.py`)
The manager spawns processes and listens to the shared queue, updating stacked `tqdm` bars flawlessly in real-time.

```python
import sys
import multiprocessing as mp
from queue import Empty
from tqdm import tqdm

# Import your training function directly
from train_rl import train_agent

def run_experiments(seeds, total_steps=1000):
    queue = mp.Queue()
    processes = []
    
    # 1. Create a tqdm bar for each seed
    bars = {}
    for i, seed in enumerate(seeds):
        bars[seed] = tqdm(
            total=total_steps, 
            position=i, 
            desc=f"Seed {seed}", 
            dynamic_ncols=True
        )
        
        # 2. Spawn processes
        p = mp.Process(target=train_agent, args=(total_steps, seed, queue))
        p.start()
        processes.append(p)

    active_jobs = len(seeds)
    last_steps = {s: 0 for s in seeds}

    # 3. Listen to the queue and update bars
    while active_jobs > 0:
        try:
            msg = queue.get(timeout=0.1)
            seed = msg.get("seed")
            
            if msg.get("done"):
                bars[seed].update(total_steps - last_steps[seed])
                bars[seed].set_description(f"Seed {seed} [DONE]")
                active_jobs -= 1
                continue
                
            # Update bar by the delta
            cur_step = msg.get("step", 0)
            bars[seed].update(cur_step - last_steps[seed])
            bars[seed].set_postfix({"loss": msg.get("loss", 0)})
            last_steps[seed] = cur_step
            
        except Empty:
            # Check if any process crashed silently
            dead_procs = [p for p in processes if not p.is_alive()]
            if len(dead_procs) > (len(seeds) - active_jobs):
                tqdm.write("Error: A worker process crashed unexpectedly.")
                break

    # 4. Clean up
    for p in processes:
        p.join()
        
    for b in bars.values():
        b.close()
        
    # Move cursor past bars
    sys.stdout.write("\n" * (len(seeds) + 1))
    sys.stdout.flush()
    print("All experiments completed.")

if __name__ == "__main__":
    mp.set_start_method('spawn') # Required for CUDA/PyTorch
    run_experiments(seeds=[1, 2, 3, 4])
```

## Bonus: Looking Forward (`rich.progress`)
If you want to move beyond `tqdm`, the Python ecosystem has largely embraced `rich` for complex terminal UIs. 

`rich.progress` allows you to render multiple progress bars alongside live tables (e.g., showing exact hyperparameter configs for each job) in a single rendering layout. It is highly recommended to migrate to `rich` if terminal dashboarding becomes a priority.

## Best Practices Summary (Updated)
1. **Never write progress to disk.** Use IPC (Inter-Process Communication) like `multiprocessing.Queue`.
2. **Set Start Method to Spawn.** When using PyTorch/CUDA in multiprocessing, always use `mp.set_start_method('spawn')` in the `__main__` block to prevent CUDA initialization deadlocks.
3. **Queue Polling.** Use `queue.get(timeout=X)` so the main thread doesn't block forever if a worker crashes without sending a `done` signal.
4. **Log to File, UI to Terminal.** Workers should configure Python's `logging` module to output to `logs_seed{X}.txt`, keeping the main terminal pristine for the `tqdm` progress bars.
