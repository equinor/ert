import json
import re
import sys
import time
from functools import lru_cache
from os import listdir, path
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, TypedDict

import numpy as np
import pandas as pd
import uvicorn
import xarray
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    DirectoryPath,
    PositiveInt,
    computed_field,
    model_validator,
    field_validator,
)
from starlette.middleware.cors import CORSMiddleware


class EnsembleSummaryChartQuery(BaseModel):
    experiment: Optional[str] = None
    ensembles: List[str]
    realizations: Union[Literal["*"], List[int]]
    selector: str


class Summary(BaseModel):
    name: str
    input_file: str
    keys: List[str]
    refcase: List[str]


class GenData(BaseModel):
    name: str
    input_file: str
    report_steps: List[int]


# When time comes, include gen_data etc but stick with only summary for now
class Response(BaseModel):
    summary: Summary
    gen_data: List[GenData]

    @computed_field
    @property
    def all_gen_data_keys(self) -> DirectoryPath:
        return [gd.name for gd in self.gen_data]


class Ensemble(BaseModel):
    id: str
    name: str
    iteration: int
    realizations: List[str]


class TransferFunctionDefinition(BaseModel):
    name: str
    distribution: str
    args: List[float]


class ParametersFile(BaseModel):
    name: str
    forward_init: bool
    template_file: Optional[str]
    transfer_function_definitions: List[TransferFunctionDefinition]
    forward_init_file: Optional[str]


class Experiment(BaseModel):
    name: str
    id: str
    ensembles: Dict[str, Ensemble]
    responses: Response
    parameters: Dict[str, ParametersFile]

    @computed_field
    @property
    def all_parameter_keys(self) -> List[str]:
        all_param_names = []

        for param_def in self.parameters.values():
            for tf_def in param_def.transfer_function_definitions:
                all_param_names.append(tf_def.name)

        return all_param_names

    def find_param_file_for_key(self, param_key: str) -> ParametersFile:
        for param_def in self.parameters.values():
            for tf_def in param_def.transfer_function_definitions:
                if tf_def.name == param_key:
                    return param_def


class WebPlotServerConfig(BaseModel):
    directory_with_ert_storage: DirectoryPath
    directory_with_html: DirectoryPath
    hostname: str
    port: PositiveInt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_tf_definition_str(tf_def_str: str) -> TransferFunctionDefinition:
    fragments = re.split(r'\s+', tf_def_str)

    return TransferFunctionDefinition(
        **{
            "name": fragments[0],
            "distribution": fragments[1],
            "args": list(
                map(float, [x for x in fragments[2:] if x != "" and x != "\n"])
            ),
        }
    )


class WebPlotStorageAccessors:
    """
    Tells the server where to access static html,
    """

    _config: WebPlotServerConfig

    def __init__(self, config: WebPlotServerConfig):
        self._config = config

    @computed_field
    @property
    def directory_with_ert_storage(self) -> DirectoryPath:
        return self._config.directory_with_ert_storage

    @computed_field
    @property
    def directory_with_ensembles(self) -> DirectoryPath:
        return self.directory_with_ert_storage / "ensembles"

    requested_experiment: str

    @computed_field
    @property
    def directory_with_experiments(self) -> DirectoryPath:
        return self.directory_with_ert_storage / "experiments"

    @lru_cache
    def get_experiments_metadata(self) -> Dict[str, Experiment]:
        experiment_ids = listdir(self.directory_with_experiments)

        experiment_infos = {}
        for exp_id in experiment_ids:
            exp_path = self.directory_with_experiments / exp_id

            name: str = exp_id
            if path.exists(exp_path / "index.json"):
                with open(exp_path / "index.json") as f:
                    exp_index = json.load(f)
                    name = exp_index["name"]

            if not path.exists(exp_path / "responses.json"):
                print(
                    f"Experiment @ {exp_path} is missing responses.json, assuming failed/stopped"
                )
                continue

            with open(exp_path / "responses.json") as f:
                responses_json = json.load(f)

            with open(exp_path / "parameter.json") as f:
                parameters_json = {
                    k: v
                    for k, v in json.load(f).items()
                    if v["_ert_kind"] == "GenKwConfig"
                }

                for param_key in parameters_json.keys():
                    param_spec = parameters_json[param_key]

                    if param_spec["_ert_kind"] != "GenKwConfig":
                        continue

                    param_spec["transfer_function_definitions"] = list(
                        map(
                            parse_tf_definition_str,
                            param_spec["transfer_function_definitions"],
                        )
                    )

            assert "summary" in responses_json, (
                f"expected 'summary' key to be found in "
                f"{exp_path / exp_id}/responses.json"
            )

            responses_in = {
                "summary": responses_json["summary"],
                "gen_data": [
                    {
                        "name": k,
                        "input_file": v["input_file"],
                        "report_steps": v["report_steps"],
                    }
                    for k, v in responses_json.items()
                    if v["_ert_kind"] == "GenDataConfig"
                ],
            }

            experiment_infos[exp_id] = Experiment(
                **{
                    "name": name,
                    "responses": responses_in,
                    "parameters": parameters_json,
                    "id": exp_id,
                    "ensembles": {},
                }
            )

        # Now read the ensembles
        ensemble_ids = listdir(self.directory_with_ensembles)
        for ens_id in ensemble_ids:
            try:
                with open(self.directory_with_ensembles / ens_id / "index.json") as f:
                    ens_index = json.load(f)

                exp_id = ens_index["experiment_id"]

                if exp_id not in experiment_infos:
                    print(f"Ensemble {ens_id} refers to invalid experiment {exp_id}")
                    continue

                iteration = ens_index["iteration"]
                realization_ids = set(
                    listdir(self.directory_with_ensembles / ens_id)
                ) - {
                    "index.json",
                    "experiment",
                }

                experiment_infos[exp_id].ensembles[ens_id] = Ensemble(
                    **{
                        "id": ens_id,
                        "iteration": int(iteration),
                        "name": ens_index["name"],
                        "realizations": realization_ids,
                    }
                )
            except FileNotFoundError:
                continue

        return experiment_infos

    def get_summary_chart_data(
        self, ensembles: List[str], experiment_id: str, keyword: str
    ):
        t0 = time.time()
        # 1. Load in all ensembles
        exp_tree = self.get_experiments_metadata()

        if experiment_id not in exp_tree:
            return {
                "MBProcessed": 0,
                "timeSpentSeconds": time.time() - t0,
                "experiment": experiment_id,
                "timeAttrs": {},
                "data": [],
                "axisX": "time",
                "axisY": "values",
            }

        # Now for the results, create line charts
        # Find one of the ensembles to determine experiment id
        requested_ensembles = []
        exp_ens_dict = exp_tree[experiment_id].ensembles
        exp_ens_vals = exp_ens_dict.values()
        for requested_ens_id in ensembles:
            if requested_ens_id == "first":
                actual_ensemble = next(
                    ens for ens in exp_ens_vals if ens.iteration == 0
                )
            elif requested_ens_id == "last":
                actual_ensemble = next(iter(exp_ens_vals))
                for ens in exp_ens_vals:
                    if ens.iteration > actual_ensemble.iteration:
                        actual_ensemble = ens
            else:
                actual_ensemble = exp_ens_dict[requested_ens_id]

            requested_ensembles.append((actual_ensemble, requested_ens_id))

        ds = None
        data = []
        total_filesize_checked = 0
        for ens, alias in requested_ensembles:
            for real in ens.realizations:
                try:
                    print(f"Selecting for ens={ens.id}, real={real}")
                    filepath = (
                        self.directory_with_ensembles
                        / ens.id
                        / f"{real}"
                        / "summary.nc"
                    )
                    ds = xarray.open_dataarray(filepath, decode_times=False)  # noqa
                    exp_tree[experiment_id].responses.summary.keys
                    total_filesize_checked += path.getsize(filepath)
                    selection = ds.sel(name=keyword)
                    values1d = selection.values.squeeze()
                    times1d = selection.coords["time"].values

                    # We normalize the points here, probably faster than doing it
                    # in the browser
                    v1dmin = values1d.min()
                    v1dmax = values1d.max()

                    t1dmin = times1d.min()
                    t1dmax = times1d.max()

                    # Think this is more vectorizable than doing the stack first
                    values1d -= v1dmin
                    values1d /= v1dmax - v1dmin

                    times1d -= t1dmin
                    times1df = times1d.astype(np.float32)
                    times1df /= t1dmax - t1dmin

                    data.append(
                        {
                            "ensemble": ens.id,
                            "ensembleAlias": alias,
                            "points": np.stack((times1df, values1d), 1).tolist(),
                            "realization": real,
                            "domainX": [int(t1dmin), int(t1dmax)],
                            "domainY": [float(v1dmin), float(v1dmax)],
                        }
                    )
                except FileNotFoundError:
                    # Missing realization, this is ok!
                    pass

        time_attrs = ds["time"].attrs if ds is not None else {}
        time_spent = time.time() - t0
        print(
            f"Spent {time_spent} seconds processing query params:"
            f"experiment={experiment_id}"
            f"ensembles={ensembles}"
            f"keyword={keyword}"
        )

        return {
            "MBProcessed": total_filesize_checked / (1000**2),
            "timeSpentSeconds": time_spent,
            "experiment": experiment_id,
            "timeAttrs": time_attrs,
            "data": data,
            "axisX": "time",
            "axisY": "values",
        }

    def get_parameter_chart_data(
        self, ensembles: List[str], experiment_id: str, parameter: str
    ):
        t0 = time.time()
        # 1. Load in all ensembles
        exp_tree = self.get_experiments_metadata()

        if experiment_id not in exp_tree:
            return {
                "MBProcessed": 0,
                "timeSpentSeconds": time.time() - t0,
                "experiment": experiment_id,
                "data": [],
            }

        # Now for the results, create line charts
        # Find one of the ensembles to determine experiment id
        requested_ensembles = []
        exp_ens_dict = exp_tree[experiment_id].ensembles
        exp_ens_vals = exp_ens_dict.values()
        for requested_ens_id in ensembles:
            if requested_ens_id == "first":
                actual_ensemble = next(
                    ens for ens in exp_ens_vals if ens.iteration == 0
                )
            elif requested_ens_id == "last":
                actual_ensemble = next(iter(exp_ens_vals))
                for ens in exp_ens_vals:
                    if ens.iteration > actual_ensemble.iteration:
                        actual_ensemble = ens
            else:
                actual_ensemble = exp_ens_dict[requested_ens_id]

            requested_ensembles.append((actual_ensemble, requested_ens_id))

        experiment = exp_tree[experiment_id]
        total_filesize_checked = 0
        param_def = experiment.find_param_file_for_key(parameter)

        if param_def is None:
            raise KeyError(f"Unknown parameter name: {parameter}")

        data_points = []
        for ens, alias in requested_ensembles:
            for real in ens.realizations:
                try:
                    filepath = (
                        self.directory_with_ensembles
                        / ens.id
                        / real
                        / f"{param_def.name}.nc"
                    )

                    ds = xarray.open_dataset(filepath)  # noqa
                    total_filesize_checked += path.getsize(filepath)

                    selection = ds.sel(names=parameter).load()
                    data_points.append(
                        {
                            "ensemble_id": ens.id,
                            "realization": real,
                            "experiment": experiment_id,
                            "values": selection["values"].values[0],
                            "transformed_values": selection[
                                "transformed_values"
                            ].values[0],
                        }
                    )

                except FileNotFoundError:
                    # Missing realization, this is ok!
                    pass

        time_spent = time.time() - t0
        print(
            f"Spent {time_spent} seconds processing query params:"
            f"experiment={experiment}"
            f"ensembles={ensembles}"
            f"parameter={parameter}"
        )

        return {
            "MBProcessed": total_filesize_checked / (1000**2),
            "timeSpentSeconds": time_spent,
            "experiment": experiment.id,
            "data": data_points,
        }

    def get_observations_data(
        self, experiment_id: str = "023b9448-fafc-4deb-a953-ff6b1ca77336"
    ):
        experiments_meta = self.get_experiments_metadata()
        experiment = experiments_meta[experiment_id]

        obs_dir = self.directory_with_experiments / experiment_id / "observations"
        observations = listdir(obs_dir)

        summaries = []
        gen_observations = []
        unclassified = []  # History observations end up here?..
        for o in observations:
            obs_dataset = xarray.load_dataset(obs_dir / o)
            response = obs_dataset.attrs["response"]

            if response == "summary":
                summaries.append(obs_dataset)
            elif response in experiment.responses.all_gen_data_keys:
                obs_dataset = obs_dataset.expand_dims(name=[response])
                gen_observations.append(obs_dataset)
            else:
                unclassified.append(obs_dataset)

        summary_obs_combined = xarray.merge(summaries)
        gen_obs_combined = xarray.merge(gen_observations)

        def ds_to_json(ds: xarray.Dataset):
            df = ds.to_dataframe().reset_index()

            if "time" in df:
                df["time"] = pd.to_datetime(df["time"]).astype(int) / 10**9

            return df.to_dict(orient="records")

        return {
            "summary": ds_to_json(summary_obs_combined),
            "gen_data": ds_to_json(gen_obs_combined),
            "experiment": experiment_id,
        }


if __name__ == "__main__":
    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": Path(sys.argv[1]),
            "directory_with_html": sys.argv[2],
            "hostname": sys.argv[3],
            "port": int(sys.argv[4]),
        }
    )

    print("Starting up web plot server...")
    print("It can be run manually by running the line below:")
    print(
        f"{path.abspath(sys.executable)} {path.abspath(__file__)} {' '.join(sys.argv[1:])}"
    )

    server_config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": Path(sys.argv[1]),
            "directory_with_html": Path(sys.argv[2]),
            "hostname": sys.argv[3],
            "port": sys.argv[4],
        }
    )
    accessors = WebPlotStorageAccessors(server_config)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/experiments")
    def read_experiments():
        return accessors.get_experiments_metadata()

    @app.get("/api/summary_chart_data")
    def get_summary_chart_data(ensembles: str, experiment: str, keyword: str):
        return accessors.get_summary_chart_data(
            ensembles.split(","), experiment, keyword
        )

    @app.get("/api/parameter_chart_data")
    def get_parameter_chart_data(ensembles: str, experiment: str, keyword: str):
        return accessors.get_parameter_chart_data(
            ensembles.split(","), experiment, parameter=keyword
        )

    @app.get("/api/observations_chart_data")
    def get_observations_data(experiment: str):
        return accessors.get_observations_data(experiment_id=experiment)

    print("Web plot API server running at...")
    print(f"{config.hostname}:{config.port}")

    app.mount("/", StaticFiles(directory=str(config.directory_with_html)))
    uvicorn.run(
        app,
        host=config.hostname,
        port=config.port,
        root_path=str(config.directory_with_html),
    )
