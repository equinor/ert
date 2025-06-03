import json
import os
from pathlib import Path

import polars as pl
import xarray as xr

from ert.storage.local_ensemble import _escape_filename

info = "Converting xr.Dataset to pl.Dataframe for genkw parameters"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        with open(experiment / "index.json", encoding="utf-8") as f:
            exp_index = json.load(f)
            experiment_id = exp_index["id"]

        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        for ens in ensembles:
            with open(ens / "index.json", encoding="utf-8") as f:
                ens_file = json.load(f)
                if ens_file["experiment_id"] != experiment_id:
                    continue

            real_dirs = [*ens.glob("realization-*")]
            for param_config in parameters_json.values():
                if param_config["_ert_kind"] == "GenKwConfig":
                    group = param_config["name"]
                    datasets = {
                        real_dir: xr.open_dataset(
                            real_dir / f"{_escape_filename(group)}.nc",
                            engine="scipy",
                        )
                        for real_dir in real_dirs
                        if (real_dir / f"{_escape_filename(group)}.nc").exists()
                    }
                    records = []
                    for real_dir, ds in datasets.items():
                        array = ds.isel(realizations=0, drop=True)["values"]
                        realization = int(real_dir.name.split("-")[1])

                        def parse_value(value: float | int | str) -> float | int | str:
                            if isinstance(value, float | int):
                                return value
                            try:
                                return int(value)
                            except ValueError:
                                try:
                                    return float(value)
                                except ValueError:
                                    return value

                        data_dict = dict(
                            zip(
                                array["names"].values.tolist(),
                                [parse_value(i) for i in array.values],
                                strict=False,
                            )
                        )
                        data_dict["realization"] = realization
                        records.append(data_dict)
                    if records:
                        df = pl.DataFrame(records).sort("realization")
                        group_path = ens / f"{_escape_filename(group)}.parquet"
                        df.write_parquet(group_path)
                        for real_dir in real_dirs:
                            if (real_dir / f"{_escape_filename(group)}.nc").exists():
                                os.remove(real_dir / f"{_escape_filename(group)}.nc")
