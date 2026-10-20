"""
experiments

The entry points of the project: the training, the evaluation and the
experiment scripts.

Each script writes into the run tree under results/:

    results/run_<name>/models/   the policy and its normalization stats
    results/run_<name>/log/      the training curve
    results/run_<name>/metrics/  the evaluation tables of that policy
    results/run_<name>/plots/    the figures of that policy

and the tables that compare several runs go into results/evaluation/.

The library these scripts drive lives in the MillingDT package.
"""
