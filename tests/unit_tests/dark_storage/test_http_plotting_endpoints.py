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
import xarray as xr
import yappi
from requests import Response

from ert.dark_storage.endpoints.plotting import set_timing_callback
from ert.storage import Storage

USE_YAPPI = os.getenv("USE_YAPPI") == "1"


def _get_kw_from_summary(
    storage: Storage,
    dark_storage_client,
    num_ensembles,
    num_keywords,
    num_timesteps,
    num_realizations,
    xarray_loading_approach: Literal["combine_nested", "mfdataset"],
    xarray_query_order: Literal["before_merge", "after_merge"],
    data_encoding_format: Literal["parquet", "arrow"],
):
    exp = storage.create_experiment()

    prev_ens = None

    start_date = np.datetime64('2010-01-01')

    timesteps = [start_date + np.timedelta64(i * 10, "D") for i in range(num_timesteps)]
    keywords = [f"kw_{i}" for i in range(num_keywords)]
    values = np.random.uniform(
        -5000, 5000, size=(1, num_keywords, num_timesteps)
    ).astype(np.float32)

    for iteration in range(num_ensembles):
        ens = exp.create_ensemble(
            ensemble_size=num_realizations,
            name=f"ens{iteration}",
            iteration=iteration,
            prior_ensemble=prev_ens,
        )

        for i_real in range(num_realizations):
            ds = xr.Dataset(
                data_vars={"values": (["realization", "name", "time"], values)},
                coords={
                    "name": keywords,
                    "time": timesteps,
                    "realization": np.arange(1, dtype=np.int32),
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
        all = pa.ipc.open_stream(reader).read_all()
        dataset = pa.Table.from_batches(stream_reader.iter_batches_with_custom_metadata())
        print("2")

    t2 = time.time()

    if USE_YAPPI:
        yappi.stop()

    timings = [
        ("dark_storage_client.get", t1 - t0),
        ("pq.read_table(parquet_buffer)", t2 - t1),
    ]

    yappi_stats = None

    if USE_YAPPI:
        yappi_stats = yappi.get_func_stats()

    return timings, yappi_stats


def test_get_kw_from_summary(fresh_storage, dark_storage_client):
    num_ensembles = 1
    num_keywords = 100
    num_timesteps = 200
    num_realizations = 200

    # Each endpoint returns the same but
    # retrieves the data in a different way
    methods = ["default"]

    inner_timings = None

    def timings_callback(times: List[Tuple[str, float]]):
        nonlocal inner_timings
        inner_timings = times

    set_timing_callback(timings_callback)

    query_configs = [
        {
            "xarray_loading_approach":load_approach,
            "xarray_query_order":query_order,
            "data_encoding_format":encoding_format,
        }
        for load_approach, query_order, encoding_format in product(
            ["combine_nested", "mfdataset"],
            ["before_merge", "after_merge"],
            ["parquet", "arrow"]
        )
    ]

    for i, query_config in enumerate(query_configs):
        outer_timings, yappi_stats = _get_kw_from_summary(
            fresh_storage,
            dark_storage_client,
            num_ensembles,
            num_keywords,
            num_timesteps,
            num_realizations,
            **query_config
        )

        base_log_dir = os.getenv("PROFILES_OUTPUT_PATH") or os.getcwd()
        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        with open(
            f"{str(base_log_dir)}/_get_kw_from_summary@{now}_config{i}", "w+"
        ) as f:
            stats_json = {
                "num_ensembles": num_ensembles,
                "num_keywords": num_keywords,
                "num_timesteps": num_timesteps,
                "num_realizations": num_realizations,
                "config": query_config,
                "timings_inner": {k: t for k, t in (inner_timings or [])},
                "timings_outer": {k: t for k, t in outer_timings},
            }

            json.dump(stats_json, f)

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
