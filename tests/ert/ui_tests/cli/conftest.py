import os

import pytest


@pytest.fixture(autouse=True)
def reduce_omp_num_threads_count():
    old_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    old_polars_max_thread = os.environ.get("POLARS_MAX_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["POLARS_MAX_THREADS"] = "1"

    yield

    if old_omp_num_threads is None:
        del os.environ["OMP_NUM_THREADS"]
    else:
        os.environ["OMP_NUM_THREADS"] = old_omp_num_threads
    if old_polars_max_thread is None:
        del os.environ["POLARS_MAX_THREADS"]
    else:
        os.environ["POLARS_MAX_THREADS"] = old_polars_max_thread
