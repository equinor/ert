import json
import os
import uuid
from pathlib import Path

import polars as pl
import pytest
import xarray as xr

from ert.storage.migration.to16 import migrate_netcdf_content_to_scalars

OLD_VERSION = 15
NEW_VERSION = 16


def setup_experiments_and_ensembles(
    experiment_uuid: uuid.UUID, ensemble_uuid: uuid.UUID, ensemble_size: int
) -> None:
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump({"version": OLD_VERSION, "migrations": []}, f, indent=2)

    os.mkdir("experiments")
    os.mkdir("ensembles")

    exp_path = Path("experiments", str(experiment_uuid))
    os.mkdir(exp_path)
    with open(Path(exp_path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": str(experiment_uuid),
                "name": "exp_0",
                "ensembles": [str(ensemble_uuid)],
            },
            f,
            indent=2,
        )

    ens_id = ensemble_uuid

    ens_path = Path("ensembles", str(ens_id))
    os.mkdir(ens_path)
    with open(Path(ens_path) / "index.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": str(ensemble_uuid),
                "experiment_id": str(experiment_uuid),
                "ensemble_size": ensemble_size,
                "iteration": 0,
                "name": "batch_0",
                "prior_ensemble_id": None,
                "started_at": "2006-10-31T14:41:41.278305",
                "everest_realization_info": {
                    "0": {"model_realization": 0, "perturbation": -1},
                },
            },
            f,
            indent=2,
        )

    for iens in range(ensemble_size):
        realization_path = ens_path / f"realization-{iens}"
        os.mkdir(realization_path)

        ds = xr.Dataset(
            {
                "values": ("names", [0.0, 0.1, -0.80]),
                "names": [x.replace(".", "\0") for x in ["x.0", "x.1", "x.2"]],
            }
        )
        ds = ds.expand_dims(realizations=[iens])
        ds.to_netcdf(realization_path / "point.nc", engine="scipy")

        ds = xr.Dataset(
            {
                "values": ("names", [0.33, 0.22 + iens]),
                "names": [column.replace(".", "\0") for column in ["x.0", "x.1"]],
            }
        )
        ds = ds.expand_dims(realizations=[iens])
        ds.to_netcdf(realization_path / "point2.nc", engine="scipy")

        ds = xr.Dataset(
            {
                "values": ("names", ["foo", "bar"]),
                "names": ["foo?", "bar!"],
            }
        )
        # Not all datasets have the realization. We should get it from path
        ds.to_netcdf(realization_path / "status.nc", engine="scipy")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_netcdf_control_value_is_added_to_scalars_parquet():
    experiment_uuid = uuid.uuid4()
    ensemble_uuid = uuid.uuid4()
    ensemble_size = 10
    setup_experiments_and_ensembles(
        experiment_uuid=experiment_uuid,
        ensemble_uuid=ensemble_uuid,
        ensemble_size=ensemble_size,
    )

    migrate_netcdf_content_to_scalars(Path(os.getcwd()))

    scalars_file = Path("ensembles", str(ensemble_uuid), "SCALAR.parquet")
    assert scalars_file.exists()
    df = pl.read_parquet(scalars_file)
    expected_df = pl.DataFrame(
        {
            "realization": list(range(ensemble_size)),
            "point.x.0": [0.0] * ensemble_size,
            "point.x.1": [0.1] * ensemble_size,
            "point.x.2": [-0.80] * ensemble_size,
            "point2.x.0": [0.33] * ensemble_size,
            "point2.x.1": [0.22 + iens for iens in range(ensemble_size)],
            "status.foo?": ["foo"] * ensemble_size,
            "status.bar!": ["bar"] * ensemble_size,
        }
    )
    df = _sort_dataframe(df)
    expected_df = _sort_dataframe(expected_df)
    assert df.columns == expected_df.columns
    assert df.dtypes == expected_df.dtypes
    assert df.equals(expected_df)
    # assert .nc files were deleted?


def _sort_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    other_cols = sorted([c for c in df.columns if c != "realization"])
    print(f"{other_cols=}")
    current_df = df.select(["realization", *other_cols]).cast({"realization": pl.Int32})
    current_df = current_df.sort(pl.col("realization"))
    return current_df
