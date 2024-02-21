import io
import os
import time
from typing import Callable, List, Literal, Tuple
from uuid import UUID

import pyarrow as pa
import xarray as xr
from fastapi import APIRouter, Body, Depends
from fastapi.responses import Response, StreamingResponse

from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["ensemble"])
DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)

_timing_callback: Callable[[List[Tuple[str, float]]], any] = lambda _: None


def set_timing_callback(callback: Callable[[List[Tuple[str, float]]], any]):
    global _timing_callback
    _timing_callback = callback


@router.get("/plotdata/summarykw/{experiment_id}/{ensemble_id}/{keyword}")
def get_summary_keyword(
    *,
    storage: Storage = DEFAULT_STORAGE,
    experiment_id: str,
    ensemble_id: str,
    keyword: str,
    xarray_loading_approach: Literal["combine_nested", "mfdataset", "single_dataset"],
    data_encoding_format: Literal["parquet", "arrow"],
    xarray_query_order: Literal["before_merge", "after_merge"],
):
    global _timing_callback
    ens = storage.get_ensemble(UUID(ensemble_id))
    assert str(ens.experiment_id) == experiment_id
    t0 = time.time()

    timings = []

    ds_paths = []
    for realization in range(ens.ensemble_size):
        input_path = ens._realization_dir(realization) / "summary.nc"
        if input_path.exists():
            ds_paths.append(input_path)

    if xarray_loading_approach == "combine_nested":
        if xarray_query_order == "after_merge":
            xrds = xr.combine_nested(
                [xr.open_dataset(p) for p in ds_paths], concat_dim="realization"
            )
            timings.append(("load dataset", time.time() - t0))
            t0 = time.time()
            xrds = xrds.sel(name=keyword, drop=True)
            timings.append(("xrd.sel(..., drop=True)", time.time() - t0))
        else:
            xrds = xr.combine_nested(
                [xr.open_dataset(p).sel(name=keyword, drop=True) for p in ds_paths],
                concat_dim="realization",
            )

            timings.append(("load & select from datasets", time.time() - t0))

    elif xarray_loading_approach == "single_dataset":
        all_ds = xr.combine_nested(
            [xr.open_dataset(p) for p in ds_paths], concat_dim="realization"
        )
        all_ds.to_netcdf(f"{ensemble_id}_combined.nc", engine="scipy")
        timings.append(("combine to one dataset", time.time() - t0))
        t0 = time.time()
        ds = xr.open_dataset(f"{ensemble_id}_combined.nc", engine="scipy")
        timings.append(("open combined dataset", time.time() - t0))
        t0 = time.time()
        xrds = ds.sel(name=keyword, drop=True)
        timings.append(("select from combined dataset", time.time() - t0))
    else:
        t0 = time.time()
        xrds = xr.open_mfdataset(
            ds_paths, combine="nested", concat_dim="realization", parallel=True
        )
        timings.append(("xr.open_mfdataset", time.time() - t0))

        t0 = time.time()
        xrds = xrds.sel(name=keyword, drop=True)
        timings.append(("mfdataset.sel(..., drop=True)", time.time() - t0))

    t1 = time.time()

    df = xrds.to_dataframe()

    t2 = time.time()
    timings.append(("xrds.to_dataframe()", t2 - t1))

    content = None
    media_type = None
    if data_encoding_format == "parquet":
        stream = io.BytesIO()
        df.to_parquet(stream)
        content = stream.getvalue()
        timings.append(("Convert to parquet stream", time.time() - t2))
        media_type = "application/x-parquet"
    else:

        async def table_generator():
            table = pa.Table.from_pandas(df)

            for batch in table.to_batches():
                serialized = batch.serialize()
                yield serialized.to_pybytes()

        if _timing_callback:
            _timing_callback(timings)

        return StreamingResponse(
            table_generator(), media_type="application/vnd.apache.arrow.file"
        )

    if _timing_callback:
        _timing_callback(timings)

    return Response(
        content=content,
        media_type=media_type,
    )
