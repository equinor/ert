import json
import sys
import time
from functools import lru_cache
from os import listdir, path
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import uvicorn
import xarray
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, DirectoryPath, PositiveInt, computed_field
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


# When time comes, include gen_data etc but stick with only summary for now
class Response(BaseModel):
    summary: Summary


class Ensemble(BaseModel):
    id: str
    name: str
    iteration: int
    realizations: List[str]


class Experiment(BaseModel):
    name: str
    id: str
    ensembles: Dict[str, Ensemble]
    responses: Response


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

            assert "summary" in responses_json, (
                f"expected 'summary' key to be found in "
                f"{exp_path / exp_id}/responses.json"
            )

            experiment_infos[exp_id] = Experiment(
                **{
                    "name": name,
                    "responses": responses_json,
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

    def get_summary_chart_data(self, ensembles: str, experiment: str, keyword: str):
        t0 = time.time()
        # 1. Load in all ensembles
        exp_tree = self.get_experiments_metadata()
        if experiment == "auto":
            tried = []
            for exp_key in exp_tree.keys():
                try:
                    return get_summary_chart_data(ensembles, exp_key, keyword)
                except Exception as e:
                    tried.append((exp_key + "\n" + str(e) + "-----------------"))

            return (
                f"Failed to find experiment with summary data,"
                f" tried the following experiments:"
                f"{''.join(tried)}"
            )

        if experiment not in exp_tree:
            return {
                "MBProcessed": 0,
                "timeSpentSeconds": time.time() - t0,
                "experiment": experiment,
                "timeAttrs": {},
                "data": [],
                "axisX": "time",
                "axisY": "values",
            }

        # Now for the results, create line charts
        # Find one of the ensembles to determine experiment id
        requested_ensembles = []
        exp_ens_dict = exp_tree[experiment].ensembles
        exp_ens_vals = exp_ens_dict.values()
        for requested_ens_id in ensembles.split(","):
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
                    exp_tree[experiment].responses.summary.keys
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
            f"experiment={experiment}"
            f"ensembles={ensembles}"
            f"keyword={keyword}"
        )

        return {
            "MBProcessed": total_filesize_checked / (1000**2),
            "timeSpentSeconds": time_spent,
            "experiment": experiment,
            "timeAttrs": time_attrs,
            "data": data,
            "axisX": "time",
            "axisY": "values",
        }


# test_query_params: EnsembleSummaryChartQuery = EnsembleSummaryChartQuery.model_validate(
#    {
#        "ensembles": ["first", "last"],
#        "realizations": "*",
#        "keyword": "FOPR",
#        "experiment": requested_experiment,
#    }
# )

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
        return accessors.get_summary_chart_data(ensembles, experiment, keyword)

    app.mount("/", StaticFiles(directory=config.directory_with_html))
    uvicorn.run(
        app,
        host=config.hostname,
        port=config.port,
        root_path=config.directory_with_html,
    )
