import json
import os
import time
from datetime import datetime
from io import BytesIO
from itertools import product
from typing import List, Literal, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
import yappi
from requests import Response

from ert.dark_storage.endpoints.plotting import set_timing_callback
from ert.storage import Storage

USE_YAPPI = os.getenv("USE_YAPPI") == "1"


def _get_kw_from_summary(
    storage: Storage,
    dark_storage_client,
    num_keywords,
    num_timesteps,
    num_realizations,
    xarray_loading_approach: Literal["combine_nested", "mfdataset", "single_dataset"],
    xarray_query_order: Literal["before_merge", "after_merge"],
    data_encoding_format: Literal["parquet", "arrow"],
):
    exp = storage.create_experiment()

    prev_ens = None

    start_date = np.datetime64('2010-01-01')

    timesteps = [start_date + np.timedelta64(i * 10, "D") for i in range(num_timesteps)]
    keywords = [f"kw_{i}" for i in range(num_keywords)]
    values = np.random.uniform(-5000, 5000, size=(num_keywords, num_timesteps)).astype(
        np.float32
    )

    for iteration in range(1):
        ens = exp.create_ensemble(
            ensemble_size=num_realizations,
            name=f"ens{iteration}",
            iteration=iteration,
            prior_ensemble=prev_ens,
        )

        for i_real in range(num_realizations):
            ds = xr.Dataset(
                data_vars={"values": (["name", "time"], values)},
                coords={
                    "name": keywords,
                    "time": timesteps,
                },
            )

            ens.save_response(group="summary", data=ds, realization=i_real)

    # Test for all ensembles
    ensids = [str(ens.id) for ens in [*exp.ensembles]]

    if USE_YAPPI:
        yappi.clear_stats()
        yappi.start()

    t0 = time.time()
    res: Response = dark_storage_client.get(
        f"/plotdata/summarykw/{str(exp.id)}/{','.join(ensids)}/kw_0",
        params={
            "xarray_loading_approach": xarray_loading_approach,
            "data_encoding_format": data_encoding_format,
            "xarray_query_order": xarray_query_order,
        },
    )
    t1 = time.time()

    if data_encoding_format == "parquet":
        parquet_buffer = BytesIO(res.content)
        pq.read_table(parquet_buffer)
    else:
        reader = pa.BufferReader(res.content)
        # TODO, check parquet performance first

    t2 = time.time()

    if USE_YAPPI:
        yappi.stop()

    timings = [t1 - t0, t2 - t1]

    yappi_stats = None

    if USE_YAPPI:
        yappi_stats = yappi.get_func_stats()

    return timings, yappi_stats


@pytest.mark.parametrize(
    "num_keywords, num_timesteps, num_realizations",
    [(10000, 400, 400), (5000, 800, 400), (5000, 200, 1600)],
)
def test_get_kw_from_summary(
    fresh_storage, dark_storage_client, num_keywords, num_timesteps, num_realizations
):
    inner_timings = None

    def timings_callback(times: List[Tuple[str, float]]):
        nonlocal inner_timings
        inner_timings = times

    set_timing_callback(timings_callback)

    query_configs = [
        {
            "xarray_loading_approach": load_approach,
            "xarray_query_order": query_order,
            "data_encoding_format": encoding_format,
        }
        for load_approach, query_order, encoding_format in product(
            ["combine_nested", "mfdataset", "single_dataset"],
            ["before_merge", "after_merge"],
            ["parquet"],
        )
    ]

    for i, query_config in enumerate(query_configs):
        outer_timings, yappi_stats = _get_kw_from_summary(
            fresh_storage,
            dark_storage_client,
            num_keywords,
            num_timesteps,
            num_realizations,
            **query_config,
        )

        base_log_dir = os.getenv("PROFILES_OUTPUT_PATH") or os.getcwd()
        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        with open(
            f"{str(base_log_dir)}/_get_kw_from_summary[{num_realizations}{num_keywords}{num_timesteps}]@{now}_config{i}",
            "w+",
        ) as f:
            [t_fetch, t_parse] = outer_timings
            t_total = t_fetch + t_parse
            t_get_inner = sum(t for _, t in inner_timings)
            t_network_overhead = t_fetch - t_get_inner

            stats_json = {
                "num_keywords": num_keywords,
                "num_timesteps": num_timesteps,
                "num_realizations": num_realizations,
                "config": query_config,
                "timings_summary": {
                    "time_total": t_total,
                    "time_network_overhead": t_network_overhead,
                    "time_fetch": t_fetch,
                    "time_parse": t_parse,
                },
                "timings_inner": {k: t for k, t in (inner_timings or [])},
            }

            json.dump(stats_json, f, indent="   ")

        if yappi_stats:
            with open(
                f"{str(base_log_dir)}/_get_kw_from_summary_yappi@{now}_{i}", "w+"
            ):
                yappi_stats.print_all(out=f)

    assert base_log_dir is not None
    print(
        f"take a look at the profile @ \n"
        "---------------------------------"
        f"{base_log_dir}"
        "---------------------------------"
    )
