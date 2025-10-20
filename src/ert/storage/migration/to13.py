import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from ert.storage.local_ensemble import _escape_filename

info = (
    "Split up one GenKw group config into single parameters. "
    "Point experiments to ensembles"
)


def point_experiments_to_ensembles(path: Path) -> None:
    experiment_to_ensemble: dict[str, list[str]] = {}
    for ensemble in path.glob("ensembles/*"):
        index_ = json.loads((ensemble / "index.json").read_text(encoding="utf-8"))
        ensemble_id = index_["id"]
        experiment_id = index_["experiment_id"]

        if experiment_id not in experiment_to_ensemble:
            experiment_to_ensemble[experiment_id] = []

        experiment_to_ensemble[experiment_id].append(ensemble_id)

    for experiment_path in path.glob("experiments/*"):
        old_json = json.loads(
            (experiment_path / "index.json").read_text(encoding="utf-8")
        )
        exp_id = old_json["id"]

        if exp_id not in experiment_to_ensemble:
            (experiment_path / "index.json").write_text(
                json.dumps(
                    old_json | {"ensembles": []},
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        (experiment_path / "index.json").write_text(
            json.dumps(
                old_json
                | {"ensembles": sorted(experiment_to_ensemble.get(exp_id, []))},
                indent=2,
            ),
            encoding="utf-8",
        )


tfd_to_distributions = {
    "NORMAL": ["name", "mean", "std"],
    "LOGNORMAL": ["name", "mean", "std"],
    "UNIFORM": ["name", "min", "max"],
    "LOGUNIF": ["name", "min", "max"],
    "TRUNCATED_NORMAL": ["name", "mean", "std", "min", "max"],
    "RAW": ["name"],
    "CONST": ["name", "value"],
    "DUNIF": ["name", "steps", "min", "max"],
    "TRIANGULAR": ["name", "min", "mode", "max"],
    "ERRF": ["name", "min", "max", "skewness", "width"],
    "DERRF": ["name", "steps", "min", "max", "skewness", "width"],
}


def migrate_gen_kw_param(parameters_json: dict[str, Any]) -> dict[str, Any]:
    new_configs = {}
    for param_config in parameters_json.values():
        if param_config["type"] == "gen_kw":
            group = param_config["name"]
            tfds = param_config["transform_function_definitions"]
            for tfd in tfds:
                # here the previous transform function definition param_name was
                # the distribution type so it needs to be provided as the first value
                # in the new distribution dict
                dist_type = tfd["param_name"]
                keys = tfd_to_distributions[dist_type]
                vals = [dist_type.lower()] + tfd["values"]
                input_source = (
                    "design_matrix"
                    if tfd["param_name"] == "RAW" and not param_config["update"]
                    else "sampled"
                )
                new_configs[tfd["name"]] = {
                    "name": tfd["name"],
                    "type": "gen_kw",
                    "group": group,
                    "distribution": dict(zip(keys, vals, strict=False)),
                    "forward_init": False,
                    "update": param_config["update"],
                    "input_source": input_source,
                }
        else:
            new_configs[param_config["name"]] = param_config
    return new_configs


def migrate_genkw(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        exp_index = json.loads((experiment / "index.json").read_text(encoding="utf-8"))
        experiment_id = exp_index["id"]

        parameters_json = json.loads(
            (experiment / "parameter.json").read_text(encoding="utf-8")
        )

        new_parameter_configs = migrate_gen_kw_param(parameters_json)
        Path(experiment / "parameter.json").write_text(
            json.dumps(new_parameter_configs, indent=2), encoding="utf-8"
        )

        # migrate parquet files
        for ens in ensembles:
            ens_file = json.loads((ens / "index.json").read_text(encoding="utf-8"))
            if ens_file["experiment_id"] != experiment_id:
                continue

            group_dfs = []
            for param_config in parameters_json.values():
                if param_config["type"] == "gen_kw":
                    group = param_config["name"]
                    group_path = ens / f"{_escape_filename(group)}.parquet"
                    if group_path.exists():
                        group_dfs.append(pl.read_parquet(group_path))
                        os.remove(group_path)
            if group_dfs:
                df = pl.concat(group_dfs, how="align")
                df = df.unique(subset=["realization"], keep="first").sort("realization")
                df.write_parquet(ens / "SCALAR.parquet")


def migrate(path: Path) -> None:
    migrate_genkw(path)
    point_experiments_to_ensembles(path)
