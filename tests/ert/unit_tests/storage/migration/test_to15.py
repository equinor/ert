import json
import os
import uuid
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from ert.storage.migration.to14 import migrate_netcdf_content_to_scalars

OLD_VERSION = 13
NEW_VERSION = 14


def setup_experiments_and_ensembles(
    experiment_uuid: uuid.UUID,
    ensemble_uuid: uuid.UUID,
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
                "ensemble_size": 1,
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
    realization_path = ens_path / "realization-0"
    os.mkdir(realization_path)
    name = "point"
    from_data = np.array([[0.33], [0.22], [-0.80]])
    ds = xr.Dataset(
        {
            "values": ("names", from_data[:, 0]),
            "names": [
                x.split(f"{name}.")[1].replace(".", "\0")
                for x in ["point.x.0", "point.x.1", "point.x.2"]
            ],
        }
    )
    ds = ds.expand_dims(realizations=[0])
    ds.to_netcdf(realization_path / "test.nc", engine="scipy")
    # print(ds.to_dataframe().reset_index().co)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_netcdf_control_value_is_added_to_scalars_parquet():
    experiment_uuid = uuid.uuid4()
    ensemble_uuid = uuid.uuid4()
    setup_experiments_and_ensembles(
        experiment_uuid=experiment_uuid, ensemble_uuid=ensemble_uuid
    )
    print(os.getcwd())

    migrate_netcdf_content_to_scalars(Path(os.getcwd()))

    scalars_file = Path("ensembles", str(ensemble_uuid), "SCALAR.parquet")
    assert scalars_file.exists()
    df = pl.read_parquet(scalars_file)
    real_column = df["realization"]
    assert len(real_column) == 1
    assert real_column[0] == 0  # Realization 0
    assert sorted(df.columns) == sorted(
        ["realization", "x\0000", "x\0001", "x\0002"]
    )  # Why are we getting double? It was supposed to be x\01 x\02


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convertion_only_appends_to_scalars_parquet_and_does_not_overwrite():
    experiment_uuid = uuid.uuid4()
    ensemble_uuid = uuid.uuid4()
    setup_experiments_and_ensembles(
        experiment_uuid=experiment_uuid, ensemble_uuid=ensemble_uuid
    )

    scalars_file = Path("ensembles", str(ensemble_uuid), "SCALAR.parquet")
    old_data = pl.DataFrame({"realization": [0], "foo": ["foo?"], "bar": ["bar!"]})
    old_data.write_parquet(scalars_file)

    migrate_netcdf_content_to_scalars(Path(os.getcwd()))

    assert scalars_file.exists()
    df = pl.read_parquet(scalars_file)
    real_column = df["realization"]
    assert len(real_column) == 1
    assert real_column[0] == 0  # Realization 0
    assert sorted(df.columns) == sorted(
        ["realization", "foo", "bar", "x\0000", "x\0001", "x\0002"]
    )
