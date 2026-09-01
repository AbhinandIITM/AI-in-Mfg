import os
import sys
import time
import subprocess
import threading
from queue import Empty, Queue
from tqdm import tqdm
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def read_progress(prog_file):
    try:
        with open(prog_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError, IndexError):
        return None

def slot_worker(slot_id, job_queue, lock, done_counter, total_jobs):
    # Initialize the progress bar for this slot
    bar = tqdm(
        total=1,
        position=slot_id,
        leave=True,
        dynamic_ncols=True,
        desc=f"slot {slot_id} idle"
    )

    while True:
        try:
            job_args = job_queue.get_nowait()
        except Empty:
            break

        algo = job_args["algo"]
        seed = job_args["seed"]
        steps = job_args["steps"]
        prog_file = str(PROJECT_ROOT / "logs" / f"progress_{algo}_seed{seed}.txt")
        
        try: os.remove(prog_file)
        except OSError: pass

        bar.reset(total=steps)
        bar.set_description(f"Running {algo} (Seed {seed})")
        bar.refresh()

        cmd = [sys.executable, str(PROJECT_ROOT / "agent" / "train_rl.py"), 
               "--algo", algo,
               "--steps", str(steps), 
               "--seed", str(seed),
               "--progress_file", prog_file]
               
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        last_n = 0
        while proc.poll() is None:
            cur = read_progress(prog_file)
            if cur is not None and cur > last_n:
                bar.update(cur - last_n)
                last_n = cur
            time.sleep(1.0)
            
        rc = proc.wait()

        if rc == 0:
            bar.update(max(0, steps - last_n))
            
        with lock:
            done_counter[0] += 1
            status = "Success" if rc == 0 else f"Failed (code {rc})"
            tqdm.write(f"  [{done_counter[0]}/{total_jobs}] {algo} Seed {seed} | {status}")

    bar.set_description(f"slot {slot_id} done")
    bar.refresh()
    bar.close()

def run_experiments(jobs, parallel_slots=4):
    q = Queue()
    for j in jobs:
        q.put(j)

    lock = threading.Lock()
    done_counter = [0]
    total_jobs = len(jobs)
    n_slots = min(parallel_slots, total_jobs)

    threads = [
        threading.Thread(
            target=slot_worker, 
            args=(i, q, lock, done_counter, total_jobs),
            daemon=True
        ) for i in range(n_slots)
    ]
    
    for t in threads: t.start()
    for t in threads: t.join()

    sys.stdout.write("\n" * (n_slots + 1))
    sys.stdout.flush()
    print("All experiments completed.")

if __name__ == "__main__":
    # Total steps per run
    steps = 200_000 
    seeds = [1, 2,3, 4, 5] 
    algos = ["PPO", "SAC"]
    
    jobs = []
    for algo in algos:
        for seed in seeds:
            jobs.append({"algo": algo, "seed": seed, "steps": steps})
            
    run_experiments(jobs=jobs, parallel_slots=8)
