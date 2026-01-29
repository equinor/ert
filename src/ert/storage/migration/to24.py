import json
import shutil
from pathlib import Path

import polars as pl

info = (
    "Move parameters.json, responses.json, observations contents to be in "
    "experiment index.json .experiment field"
)


def migrate_parameters_responses_and_observations_into_experiment_index(
    path: Path,
) -> None:
    for experiment_path in path.glob("experiments/*"):
        experiment_json = {}
        with open(experiment_path / "metadata.json", encoding="utf-8") as fin:
            metadata_json = json.load(fin)
            if "weights" in metadata_json:
                experiment_json["weights"] = metadata_json["weights"]

        responses_contents = json.loads(
            (experiment_path / "responses.json").read_text(encoding="utf-8")
        )
        parameters_contents = json.loads(
            (experiment_path / "parameter.json").read_text(encoding="utf-8")
        )

        experiment_json["response_configuration"] = list(responses_contents.values())
        experiment_json["parameter_configuration"] = list(parameters_contents.values())

        # PS: This may be super slow for large observation datasets
        # Revisit later and consider keeping the files
        # Convert existing parquet observation files into a flat list of
        # observation declarations (one declaration per row). This will be
        # stored in experiment_json["observations"].
        observations_list: list[dict[str, object]] = []
        if (experiment_path / "observations").exists():
            for path_to_obs_file in (experiment_path / "observations").glob("*"):
                response_type = path_to_obs_file.stem
                df = pl.read_parquet(path_to_obs_file)
                for row in df.to_dicts():
                    if response_type == "summary":
                        # Expect columns: response_key, observation_key, time, observations, std, east, north, radius
                        time_val = row.get("time")
                        date_str = (
                            time_val.date().isoformat()
                            if hasattr(time_val, "date")
                            else str(time_val)
                        )
                        decl = {
                            "type": "summary_observation",
                            "name": row.get("observation_key"),
                            "value": float(row.get("observations"))
                            if row.get("observations") is not None
                            else None,
                            "error": float(row.get("std"))
                            if row.get("std") is not None
                            else None,
                            "key": row.get("response_key"),
                            "date": date_str,
                        }
                        # localization
                        if row.get("east") is not None or row.get("north") is not None:
                            localization = {}
                            if row.get("east") is not None:
                                localization["east"] = float(row.get("east"))
                            if row.get("north") is not None:
                                localization["north"] = float(row.get("north"))
                            if row.get("radius") is not None:
                                localization["radius"] = float(row.get("radius"))
                            decl.update(localization)
                        observations_list.append(decl)
                    elif response_type == "gen_data":
                        # Expect columns: response_key, observation_key, report_step, index, observations, std
                        decl = {
                            "type": "general_observation",
                            "name": row.get("observation_key"),
                            "data": row.get("response_key"),
                            "value": float(row.get("observations"))
                            if row.get("observations") is not None
                            else None,
                            "error": float(row.get("std"))
                            if row.get("std") is not None
                            else None,
                            "restart": int(row.get("report_step"))
                            if row.get("report_step") is not None
                            else 0,
                            "index": int(row.get("index"))
                            if row.get("index") is not None
                            else 0,
                        }
                        observations_list.append(decl)

            experiment_json["observations"] = observations_list

        with open(experiment_path / "index.json", encoding="utf-8") as fin:
            index_json = json.load(fin)
            index_json["experiment"] = experiment_json

            Path(experiment_path / "index.json").write_text(
                json.dumps(index_json, indent=2), encoding="utf-8"
            )

        (experiment_path / "metadata.json").unlink()
        (experiment_path / "responses.json").unlink()
        (experiment_path / "parameter.json").unlink()

        if (experiment_path / "observations").exists():
            shutil.rmtree(experiment_path / "observations")


def migrate(path: Path) -> None:
    migrate_parameters_responses_and_observations_into_experiment_index(path)
