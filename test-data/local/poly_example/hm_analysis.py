#!/usr/bin/env python
from http.client import responses
import logging
import sys
import json
import requests
import io
import pandas as pd
import numpy as np
import datetime
import dateutil

from typing import List, Optional, Union
from ert_shared.services import Storage
from ert.analysis import ies

logger = logging.getLogger(__name__)


def _get_from_server(
    url, headers={}, params=None, status_code=200
) -> requests.Response:
    session = Storage.session()
    resp = session.get(url, headers=headers, params=params)

    if resp.status_code != status_code:
        logger.error(f"Failed to fetch from {url}. Response: {resp.text}")
    return resp


def _post_to_server(
    url, data=None, params=None, json=None, headers={}, status_code=200
) -> requests.Response:
    session = Storage.session()
    resp = session.post(
        url,
        headers=headers,
        params=params,
        data=data,
        json=json,
    )
    if resp.status_code != status_code:
        logger.error(f"Failed to post to {url}. Response: {resp.text}")

    return resp


def get_ensemble_record_data(
    ensemble_id: str,
    record_name: str,
) -> pd.DataFrame:

    resp = _get_from_server(
        url=f"ensembles/{ensemble_id}/records/{record_name}",
        headers={"accept": "application/x-parquet"},
    )
    stream = io.BytesIO(resp.content)
    df = pd.read_parquet(stream).transpose()

    try:
        df.index = df.index.astype(int)
    except TypeError:
        pass
    df = df.sort_index()
    return df


def get_ensemble_parameter_data(
    ensemble_id: str,
    parameter_name: str,
) -> pd.DataFrame:

    if "::" in parameter_name:
        name, label = parameter_name.split("::", 1)
        params = {"label": label}
    else:
        name = parameter_name
        params = {}

    resp = _get_from_server(
        url=f"ensembles/{ensemble_id}/records/{name}",
        headers={"accept": "application/x-parquet"},
        params=params,
    )
    stream = io.BytesIO(resp.content)
    df = pd.read_parquet(stream).transpose()
    df.columns = [int(c) for c in df.columns]

    return df.sort_index(axis=1)


def get_ensemble_record_observations(ensemble_id: str, record_name: str) -> List[dict]:
    return _get_from_server(
        url=f"ensembles/{ensemble_id}/records/{record_name}/observations",
        # Hard coded to zero, as all realizations are connected to the same observations
        params={"realization_index": 0},
    ).json()


def observation_data_to_df(observation) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "values": observation["values"],
            "std": observation["errors"],
            "x_axis": observation["x_axis"],
            "active": [True for _ in observation["x_axis"]],
        }
    )


def indexes_to_axis(
    indexes: List[Union[int, str, datetime.datetime]]
) -> List[Union[int, str, datetime.datetime]]:
    try:
        if indexes and type(indexes[0]) is str and not str(indexes[0]).isnumeric():
            return list(map(lambda dt: dateutil.parser.isoparse(str(dt)), indexes))
        return [int(i) for i in indexes]
    except ValueError as e:
        raise ValueError("Could not parse indexes as either int or dates", e)


def load_update_data(ensemble_id: str) -> tuple:

    responses = _get_from_server(url=f"ensembles/{ensemble_id}/responses").json()
    response_dfs = []
    combined_observation_values = []
    combined_observation_errors = []

    for key in responses:
        observations = get_ensemble_record_observations(
            ensemble_id=ensemble_id, record_name=key
        )
        df = get_ensemble_record_data(ensemble_id=ensemble_id, record_name=key)
        # this will not work for multiple observations with overlapping x-axis
        obs_axis = []
        for obs in observations:
            combined_observation_values += obs["values"]
            combined_observation_errors += obs["errors"]
            axis = indexes_to_axis(obs["x_axis"])
            obs_axis += axis

        response_dfs.append(df.loc[obs_axis])

    parameters = _get_from_server(url=f"ensembles/{ensemble_id}/parameters").json()
    parameter_dfs = [
        get_ensemble_parameter_data(
            ensemble_id=ensemble_id, parameter_name=parameter["name"]
        )
        for parameter in parameters
    ]

    return (
        pd.concat(parameter_dfs),
        pd.concat(response_dfs),
        np.array(combined_observation_values),
        np.array(combined_observation_errors),
    )


def create_ensemble(name, parameter_matrix, original_ensemble_id):
    update_create = dict(
        observation_transformations=[],
        ensemble_reference_id=original_ensemble_id,
        ensemble_result_id=None,
        algorithm="Ensemble smoother",
    )

    update = (
        _post_to_server(
            "updates",
            json=update_create,
        ).json
        | ()
    )
    experiment_id = update["experiment_id"]

    parameters = _get_from_server(
        url=f"ensembles/{original_ensemble_id}/parameters"
    ).json()
    ensemble_new = dict(
        size=len(parameter_matrix[0]),
        parameter_names=[p["name"] for p in parameters],
        response_names=list(responses.keys()),
        update_id=update["id"],
        userdata={"name": name},
        active_realizations=[],
    )
    ensemble = _post_to_server(
        f"experiments/{experiment_id}/ensembles", json=ensemble_new
    ).json()

    for index, param in enumerate(parameters):
        df = pd.DataFrame(parameter_matrix[index])
        _post_to_server(
            f"ensembles/{ensemble['id']}/records/{param['name']}/matrix",
            data=df.to_csv(),
            headers={"content-type": "text/csv"},
        )


def run_analysis(ensemble_id: str):
    parameters, responses, observation_values, observation_errors = load_update_data(
        ensemble_id
    )
    A = parameters.to_numpy()
    S = responses.to_numpy()

    noise = np.random.rand(*S.shape)
    E = ies.makeE(observation_errors, noise)
    R = np.identity(len(observation_errors))
    D = ies.makeD(observation_values, E, S)

    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    S = (S.T / observation_errors).T

    X = ies.initX(S, R, E, D)
    new_A = A @ X

    create_ensemble("smooth", new_A, ensemble_id)


if __name__ == "__main__":
    run_analysis(sys.argv[1])
