import os
import sys
import time
from multiprocessing import Process, Queue
from tqdm import tqdm
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.train_rl import train

def worker_wrapper(worker_id, algo, seed, steps, queue):
    """
    Wrapper to run the train function in a separate process.
    """
    try:
        # Redirect stdout/stderr to a log file for each worker
        log_dir = PROJECT_ROOT / "results" / "workers"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Override stdout/stderr
        sys.stdout = open(log_dir / f"worker_{algo}_{seed}.log", "w")
        sys.stderr = sys.stdout

        train(algo=algo, timesteps=steps, seed=seed, queue=queue)
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        import traceback
        traceback.print_exc()

def run_experiments(jobs, parallel_slots=4):
    total_jobs = len(jobs)
    done_counter = 0

    # Progress bars for each slot
    # Note: On Windows, multiple tqdm bars in processes can be tricky.
    # We'll use a single master bar and slot status messages.
    master_bar = tqdm(total=total_jobs, desc="Total Progress", position=0)

    active_processes = {} # {slot_id: (process, queue, job_info, start_time)}
    job_queue = jobs.copy()

    slots_bars = [tqdm(total=100, desc=f"Slot {i} idle", position=i+1, leave=True) for i in range(parallel_slots)]

    last_update = {} # {slot_id: last_timestamp}

    try:
        while done_counter < total_jobs or active_processes:
            # Fill empty slots
            while len(active_processes) < parallel_slots and job_queue:
                # Find an empty slot index
                slot_id = None
                for i in range(parallel_slots):
                    if i not in active_processes:
                        slot_id = i
                        break

                if slot_id is None: break

                job = job_queue.pop(0)
                q = Queue()
                p = Process(target=worker_wrapper, args=(slot_id, job['algo'], job['seed'], job['steps'], q))
                p.start()

                active_processes[slot_id] = (p, q, job, time.time())
                slots_bars[slot_id].reset(total=job['steps'])
                slots_bars[slot_id].set_description(f"{job['algo']} S{job['seed']}")
                last_update[slot_id] = time.time()

            # Check for updates and completion
            to_remove = []
            for slot_id, (p, q, job, start_time) in active_processes.items():
                # Check for progress updates
                while not q.empty():
                    try:
                        msg = q.get_nowait()
                        steps = msg['timesteps']
                        slots_bars[slot_id].n = steps
                        slots_bars[slot_id].refresh()
                        last_update[slot_id] = time.time()
                    except:
                        break

                # Check if process is done
                if not p.is_alive():
                    p.join()
                    done_counter += 1
                    master_bar.update(1)
                    slots_bars[slot_id].set_description(f"Slot {slot_id} idle")
                    slots_bars[slot_id].n = 0
                    slots_bars[slot_id].refresh()
                    to_remove.append(slot_id)
                    tqdm.write(f"  [{done_counter}/{total_jobs}] {job['algo']} Seed {job['seed']} | Success")
                else:
                    # Stalled worker detection (5 mins)
                    if time.time() - last_update[slot_id] > 300:
                        tqdm.write(f"  [WARN] Slot {slot_id} ({job['algo']} S{job['seed']}) stalled. Terminating.")
                        p.terminate()
                        p.join()
                        # Re-add to job queue if needed, or just mark as failed
                        done_counter += 1
                        master_bar.update(1)
                        to_remove.append(slot_id)

            for slot_id in to_remove:
                del active_processes[slot_id]
                del last_update[slot_id]

            time.sleep(0.5)

    finally:
        for p, q, job, start_time in active_processes.values():
            if p.is_alive():
                p.terminate()
                p.join()
        for bar in slots_bars:
            bar.close()
        master_bar.close()

    print("All experiments completed.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() # For Windows

    # Total steps per run
    steps = 200_000
    seeds = [1, 2, 3, 4, 5]
    algos = ["PPO"]


    jobs = []
    for algo in algos:
        for seed in seeds:
            jobs.append({"algo": algo, "seed": seed, "steps": steps})

    run_experiments(jobs=jobs, parallel_slots=min(8, multiprocessing.cpu_count()))
