import os

if "PYTEST_XDIST_WORKER" in os.environ:  # parallel workers: split the cores, set before numpy/jax
    threads = str(max(1, os.cpu_count() // int(os.environ["PYTEST_XDIST_WORKER_COUNT"])))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, threads)

from pathlib import Path

from jaxmodal import config

collect_ignore = ["multidevice"]

config.enable_x64()
config.enable_cache(Path(__file__).parents[1] / ".jax_cache")
