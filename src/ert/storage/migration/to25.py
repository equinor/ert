import datetime
import json
import shutil
from pathlib import Path
from typing import Any, cast

import polars as pl

info = (
    "Move parameters.json, responses.json, observations contents to be in "
    "experiment index.json .experiment field"
)


def _observation_row_to_declaration(
    row: dict[str, Any], response_type: str
) -> dict[str, Any]:
    observation, error = row.get("observations"), row.get("std")
    decl = {
        "name": row.get("observation_key"),
        "value": float(observation) if observation is not None else None,
        "error": float(error) if error is not None else None,
    }
    if response_type == "summary":
        date_str = cast(datetime.datetime, row.get("time")).date().isoformat()
        decl.update(
            {
                "type": "summary_observation",
                "key": row.get("response_key"),
                "date": date_str,
            }
        )
        if row.get("east") is not None or row.get("north") is not None:
            for f in ["east", "north", "radius"]:
                if (val := row.get(f)) is not None:
                    decl[f] = float(val)
    elif response_type == "gen_data":
        decl.update(
            {
                "type": "general_observation",
                "data": row.get("response_key"),
                "restart": int(row.get("report_step") or 0),
                "index": int(row.get("index") or 0),
            }
        )
    return decl


def migrate_parameters_responses_and_observations_into_experiment_index(
    path: Path,
) -> None:
    for exp_path in sorted(path.glob("experiments/*")):
        if not exp_path.is_dir():
            continue

        meta = json.loads((exp_path / "metadata.json").read_text(encoding="utf-8"))
        resps = json.loads((exp_path / "responses.json").read_text(encoding="utf-8"))
        params = json.loads((exp_path / "parameter.json").read_text(encoding="utf-8"))

        experiment_json = {
            "response_configuration": list(resps.values()),
            "parameter_configuration": list(params.values()),
            "observations": [],
        }
        if "weights" in meta:
            experiment_json["weights"] = meta["weights"]

        obs_dir = exp_path / "observations"
        if obs_dir.exists():
            for obs_file in sorted(obs_dir.glob("*")):
                df = pl.read_parquet(obs_file)
                experiment_json["observations"].extend(
                    [
                        _observation_row_to_declaration(row, obs_file.stem)
                        for row in df.to_dicts()
                    ]
                )

        index_file = exp_path / "index.json"
        index = json.loads(index_file.read_text(encoding="utf-8"))
        index["experiment"] = experiment_json
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

        for f in ["metadata.json", "responses.json", "parameter.json"]:
            (exp_path / f).unlink()
        if obs_dir.exists():
            shutil.rmtree(obs_dir)


def migrate(path: Path) -> None:
    migrate_parameters_responses_and_observations_into_experiment_index(path)
