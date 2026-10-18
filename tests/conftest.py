import os

if "PYTEST_XDIST_WORKER" in os.environ:  # split the cores for BLAS (before numpy); XLA ignores it
    threads = str(max(1, os.cpu_count() // int(os.environ["PYTEST_XDIST_WORKER_COUNT"])))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, threads)

from pathlib import Path

from jaxmodal import config

collect_ignore = ["multidevice"]

config.enable_x64()
config.enable_cache(Path(__file__).parents[1] / ".jax_cache")

LONG = (
    "test_mac_acceptance",
    "test_mac_growth_peak",
    "test_count_peak",
    "test_mac_count_peak",
    "test_estimator_large",
    "test_mac_slice_peak",
)


def pytest_collection_modifyitems(items):  # longest first, so parallel runs end without a long tail
    items.sort(key=lambda i: (i.originalname not in LONG, i.get_closest_marker("slow") is None))
