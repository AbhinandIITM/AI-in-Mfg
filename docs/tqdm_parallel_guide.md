# Multi-Process Progress Tracking with `tqdm`

Standard `tqdm` loops can easily break your terminal output when multiple processes print simultaneously. The most robust way to track parallel jobs without garbled terminal output is the **Slot-Worker Pattern** using progress files.

## Why this approach is effective
1. **No Process Locks Needed**: Standard multiprocessing `tqdm` requires complicated locks or queue mechanisms. This uses simple disk I/O which is universally compatible and non-blocking.
2. **Crash Resilience**: If a process crashes, the parent script knows exactly how many steps it completed before failing.
3. **Clean UI**: Uses `tqdm(position=slot)` to assign each running job a fixed row on your terminal, keeping bars beautifully stacked.

## The Child Script (The Worker Loop)
Your actual worker script should be completely unaware of the complex parallel UI. It just needs to periodically write its current step to a text file.

```python
# train_rl.py
import argparse

def train_agent(total_steps, progress_file=None):
    for step in range(1, total_steps + 1):
        # ... Do work here ...
        
        # Periodically write progress to a file (e.g., every 1000 steps)
        if progress_file and step % 1000 == 0:
            with open(progress_file, "w") as f:
                f.write(str(step))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--progress_file", type=str, default=None)
    args = parser.parse_args()
    
    train_agent(args.steps, args.progress_file)
```

## The Parent Script (The Manager)
The manager script creates a pool of thread "slots". Each slot runs the child script as a subprocess and monitors its progress file to update a dedicated `tqdm` bar.

```python
import os, sys, time, subprocess, threading
from queue import Empty, Queue
from tqdm import tqdm

def read_progress(prog_file):
    """Safely return current step count from worker, or None if transient failure."""
    try:
        with open(prog_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError, IndexError):
        return None

def slot_worker(slot_id, job_queue, total_steps, lock, done_counter, total_jobs):
    # Initialize the progress bar for this slot
    bar = tqdm(
        total=total_steps,
        position=slot_id,        # Fixes the bar to a specific terminal row
        leave=True,
        dynamic_ncols=True,
        desc=f"slot {slot_id} idle"
    )

    while True:
        try:
            job_args = job_queue.get_nowait()
        except Empty:
            break

        # Setup paths for this specific job
        seed = job_args["seed"]
        prog_file = f"progress_seed{seed}.txt"
        
        # Clear out old progress files if they exist
        try: os.remove(prog_file)
        except OSError: pass

        # Reset the bar for the new job
        bar.reset(total=total_steps)
        bar.set_description(f"Running Seed {seed}")
        bar.refresh()

        # Start the subprocess
        cmd = [sys.executable, "train_rl.py", 
               "--steps", str(total_steps), 
               "--progress_file", prog_file]
               
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Poll the progress file while the subprocess is running
        last_n = 0
        while proc.poll() is None:
            cur = read_progress(prog_file)
            if cur is not None and cur > last_n:
                bar.update(cur - last_n)
                last_n = cur
            time.sleep(1.0) # Check every second
            
        rc = proc.wait()

        # Job finished: Fill the bar to 100% on success
        if rc == 0:
            bar.update(max(0, total_steps - last_n))
            
        # Safely log completion using tqdm.write (prevents breaking the bars)
        with lock:
            done_counter[0] += 1
            status = "Success" if rc == 0 else f"Failed (code {rc})"
            tqdm.write(f"  [{done_counter[0]}/{total_jobs}] Seed {seed} | {status}")

    bar.set_description(f"slot {slot_id} done")
    bar.refresh()
    bar.close()

def run_experiments(seeds, parallel_slots=4, total_steps=1000000):
    q = Queue()
    for s in seeds:
        q.put({"seed": s})

    lock = threading.Lock()
    done_counter = [0]
    total_jobs = len(seeds)
    n_slots = min(parallel_slots, total_jobs)

    # Spawn the threads
    threads = [
        threading.Thread(
            target=slot_worker, 
            args=(i, q, total_steps, lock, done_counter, total_jobs),
            daemon=True
        ) for i in range(n_slots)
    ]
    
    for t in threads: t.start()
    for t in threads: t.join()

    # Move cursor past the stacked bars before printing final logs
    sys.stdout.write("\n" * (n_slots + 1))
    sys.stdout.flush()
    print("All experiments completed.")

if __name__ == "__main__":
    run_experiments(seeds=[1, 2, 3, 4, 5, 6, 7, 8], parallel_slots=4)
```

## Best Practices Summary
1. **Always use `tqdm.write()`**: If your worker threads or parent script need to print logs, NEVER use `print()`. Always use `tqdm.write()`. This prints the message neatly above the progress bars without overwriting them.
2. **Move the cursor at the end**: At the end of execution, `sys.stdout.write("\n" * (n_slots + 1))` pushes the terminal cursor completely below the stacked parallel bars so that subsequent console output doesn't overwrite your final bars.
3. **Limit I/O**: Don't let your child script write to the progress file every single step. Update it every `N` steps (e.g. 1000 or 5000) to avoid disk bottlenecking.
