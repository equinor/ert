import json
import os
from pathlib import Path

import polars as pl

from ert.storage.local_ensemble import _escape_filename

info = "Union of all GenKw parameters into a single GenKwConfig"


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

            datasets = {}
            parameters = {}
            one_gen_kw = {}
            one_gen_kw["name"] = "SCALAR"
            one_gen_kw["transform_function_definitions"] = []
            one_gen_kw["_ert_kind"] = "GenKwConfig"
            one_gen_kw["forward_init"] = False
            one_gen_kw["update"] = False
            for param_name, param_config in parameters_json.items():
                if param_config["_ert_kind"] == "GenKwConfig":
                    group = param_config["name"]
                    for tfd in param_config.get("transform_function_definitions", []):
                        new_tfd = tfd.copy()
                        new_tfd["group_name"] = group
                        new_tfd["input_source"] = "sampled"
                        new_tfd["update"] = param_config.get("update", False)
                        one_gen_kw["transform_function_definitions"].append(new_tfd)
                    tfd = {}
                    if (ens / f"{_escape_filename(group)}.parquet").exists():
                        df = pl.read_parquet(ens / f"{_escape_filename(group)}.parquet")
                        if datasets:
                            datasets[group] = df.drop("realization")
                        else:
                            datasets[group] = df
                        os.remove(ens / f"{_escape_filename(group)}.parquet")
                else:
                    parameters[param_name] = param_config
            if datasets:
                df_scalar = pl.concat(
                    [
                        datasets[group]
                        for group in datasets
                        if not datasets[group].is_empty()
                    ],
                    how="horizontal",
                )
                df_scalar.write_parquet(ens / "SCALAR.parquet")
                with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
                    parameters["SCALAR"] = one_gen_kw
                    parameters = dict(sorted(parameters.items()))
                    fout.write(json.dumps(parameters, indent=4))
