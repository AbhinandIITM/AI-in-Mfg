"""
experiments/run_experiments.py

Parallel training supervisor for the ablation study.

This module uses multiprocessing (not subprocess) to run a pool of training
workers because:
- The workers share the same Python environment and imports; spawning N
  subprocesses would re-import torch and the environment N times, adding
  ~30s overhead per worker.
- The heartbeat queue (multiprocessing.Queue) gives the supervisor a
  low-latency liveness signal from each worker; a subprocess would need a
  pipe or file watcher.
- The workers are short-lived (100k-150k steps, ~2-3 minutes each) and the
  pool size is small (4-7 slots); multiprocessing is the lighter-weight
  choice.

run_pipeline.py is the sequential orchestrator that calls *this* script as
one stage (the ablation sweep). The two strategies exist because
run_pipeline.py sequences dependent stages, while run_experiments.py
parallelizes independent jobs within one stage. Merging them would conflate
sequential orchestration with parallel worker supervision.

The job list is built by the caller: experiments/train_rl_ablation.py owns the
energy weight grid and hands one job per (weight, seed) pair to
run_experiments().
"""

import os
import sys
import time
from multiprocessing import Process, Queue
from queue import Empty
from tqdm import tqdm
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_rl import train
from MillingDT.config import WORKERS_DIR, run_models

def worker_wrapper(worker_id, job, queue):
    """
    Wrapper to run the train function in a separate process.
    """
    try:
        # Redirect stdout/stderr to a log file for each worker, so the
        # supervisor's progress bars are not buried under the training output.
        log_dir = WORKERS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        tag = job.get("tag", job["algo"])
        log_path = log_dir / f"worker_{tag}_{job['algo']}_{job['seed']}.log"
        try:
            sys.stdout = open(log_path, "w")
            sys.stderr = sys.stdout
        except OSError as exc:
            # A log that cannot be opened must not take the training down.
            print(f"Worker {worker_id} could not open {log_path}: {exc}")

        train(
            algo=job["algo"],
            timesteps=job["steps"],
            seed=job["seed"],
            queue=queue,
            env_kwargs=job.get("env_kwargs"),
            run_name=job.get("run_name"),
            randomize=job.get("randomize", False),
            randomize_weight=job.get("randomize_weight", False),
            w2=job.get("w2"),
        )
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        import traceback
        traceback.print_exc()
        # The supervisor reads the exit code to report the job as failed. A
        # worker that swallows the error would look like a job that ran.
        sys.exit(1)

def missing_policy(job):
    """The path of the policy a job was supposed to produce, when it is absent."""
    run_name = job.get("run_name")
    if not run_name:
        return None
    policy = run_models(run_name) / "final_model.zip"
    return policy if not policy.exists() else None

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
                p = Process(target=worker_wrapper, args=(slot_id, job, q))
                p.start()

                active_processes[slot_id] = (p, q, job, time.time())
                slots_bars[slot_id].reset(total=job['steps'])
                slots_bars[slot_id].set_description(f"{job['algo']} S{job['seed']}")
                last_update[slot_id] = time.time()

            # Check for updates and completion
            to_remove = []
            for slot_id, (p, q, job, start_time) in active_processes.items():
                # Drain the heartbeat queue. q.empty() is unreliable on
                # multiprocessing queues, so drain until get_nowait raises Empty.
                while True:
                    try:
                        msg = q.get_nowait()
                    except Empty:
                        break
                    steps = msg['timesteps']
                    slots_bars[slot_id].n = steps
                    slots_bars[slot_id].refresh()
                    last_update[slot_id] = time.time()

                # Check if process is done
                if not p.is_alive():
                    p.join()
                    done_counter += 1
                    master_bar.update(1)
                    slots_bars[slot_id].set_description(f"Slot {slot_id} idle")
                    slots_bars[slot_id].n = 0
                    slots_bars[slot_id].refresh()
                    to_remove.append(slot_id)
                    # The exit code alone does not tell the whole story: the
                    # job is only done when the policy it was supposed to
                    # train is on disk.
                    missing = missing_policy(job)
                    if p.exitcode == 0 and missing is None:
                        tqdm.write(f"  [{done_counter}/{total_jobs}] {job['algo']} Seed {job['seed']} | Success")
                    else:
                        reason = f"exit code {p.exitcode}" if missing is None else f"no policy at {missing}"
                        tqdm.write(f"  [{done_counter}/{total_jobs}] {job['algo']} Seed {job['seed']} | FAILED ({reason})")
                else:
                    # Stalled worker detection (5 mins without a heartbeat)
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
