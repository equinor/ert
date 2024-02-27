import json
import os
import time
from datetime import datetime
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
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


@pytest.mark.parametrize(
    "num_keywords, num_timesteps, num_realizations",
    [
        (10000, 400, 400), (5000, 800, 400), (5000, 200, 1600),
    ],
)
def test_xarray_combine(
    fresh_storage, dark_storage_client, num_keywords, num_timesteps, num_realizations
):
    exp = fresh_storage.create_experiment()

    prev_ens = None

    start_date = np.datetime64('2010-01-01')

    timesteps = [start_date + np.timedelta64(i * 10, "D") for i in range(num_timesteps)]
    keywords = [f"kw_{i}" for i in range(num_keywords)]
    values = np.random.uniform(-5000, 5000, size=(num_keywords, num_timesteps)).astype(
        np.float32
    )

    ens = None
    ds_paths = []
    time_to_save_original = 0
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

            t0 = time.time()
            if "realization" not in ds.dims:
                ds = ds.expand_dims({"realization": [i_real]})

            output_path = ens._realization_dir(i_real)
            Path.mkdir(output_path, parents=True, exist_ok=True)
            ds_paths.append(output_path / "summary.nc")

            ds.to_netcdf(output_path / "summary.nc", engine="scipy")
            time_to_save_original += time.time() - t0

    time_to_save_append = 0
    time_to_make_zarr = 0
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

            t0 = time.time()
            if "realization" not in ds.dims:
                ds = ds.expand_dims({"realization": [i_real]})

            ds.to_netcdf(
                fresh_storage.path / f"summary.nc",
                engine="scipy",
                mode=("a" if i_real > 0 else "w"),
            )
            time_to_save_append += time.time() - t0

            t0 = time.time()
            ds.to_zarr(
                fresh_storage.path / "summary.zarr",
                mode="a" if i_real > 0 else "w"
            )
            time_to_make_zarr += time.time() - t0

    time_to_combine_after = 0
    t0 = time.time()
    opened_datasets = [xr.open_dataset(p) for p in ds_paths]

    all_ds = xr.combine_nested(
        opened_datasets, concat_dim="realization"
    )
    all_ds.to_netcdf(f"{ens.id}_combined.nc", engine="scipy")
    time_to_combine_after = time.time() - t0

    t0 = time.time()
    nested = xr.combine_nested(opened_datasets, concat_dim="realization")
    nested.to_netcdf(fresh_storage.path / "summarycombined.nc", engine="scipy")
    time_to_combine_when_open = time.time() - t0

    base_log_dir = os.getenv("PROFILES_OUTPUT_PATH") or os.getcwd()
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    t0 = time.time()
    ds_zarr = xr.open_dataset(fresh_storage.path / "summary.zarr")
    time_to_open_zarr = time.time() - t0
    t0 = time.time()
    ds_zarr.to_netcdf(fresh_storage.path / "summary_zarr.nc")
    time_to_zarr_to_netcdf = time.time() - t0


    with open(
        f"{str(base_log_dir)}/test_xarray_combine.json"
        f"[{num_realizations}{num_keywords}{num_timesteps}]@{now}",
        "w+",
    ) as f:
        json.dump(
            {
                "write one file per real": time_to_save_original,
                "append to big file, once per real": time_to_save_append,
                "combine single files (excluding single file creation)": time_to_combine_after,
                "combine when ds' are open": time_to_combine_when_open,
                "time_to_make_zarr": time_to_make_zarr,
                "time_to_open_zarr": time_to_open_zarr,
                "time_to_zarr_to_netcdf": time_to_zarr_to_netcdf,
            },
            f,
            indent="    "
        )
