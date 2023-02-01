import datetime
import io
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Union

import pandas as pd
import requests

from ert._c_wrappers.enkf.enums.enkf_obs_impl_type_enum import (
    EnkfObservationImplementationType,
)
from ert.data import MeasuredData
from ert.services import Storage
from ert.shared.feature_toggling import feature_enabled

if TYPE_CHECKING:
    from ert.libres_facade import LibresFacade

logger = logging.getLogger()


def create_experiment(ert) -> dict:
    return dict(name=str(datetime.datetime.now()), priors=create_priors(ert))


def create_ensemble(
    size: int,
    case_name: str,
    active_realizations: List[int],
    parameter_names: List[str],
    response_names: List[str],
    update_id: str = None,
) -> dict:
    return dict(
        size=size,
        parameter_names=parameter_names,
        response_names=response_names,
        update_id=update_id,
        userdata={"name": case_name},
        active_realizations=active_realizations,
    )


def create_parameters(ert) -> List[dict]:
    parameters = [
        dict(
            name=key,
            values=list(parameter.values),
        )
        for key, parameter in (
            (key, ert.gather_gen_kw_data(ert.get_current_case_name(), key))
            for key in ert.all_data_type_keys()
            if ert.is_gen_kw_key(key)
        )
    ]

    return parameters


def _create_response_observation_links(ert) -> Mapping[str, str]:
    observation_vectors = ert.get_observations()
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    if keys == []:
        return {}

    data = MeasuredData(ert, keys, load_data=False)
    observations = data.data.loc[["OBS", "STD"]]
    response_observation_link = {}

    for obs_key in observations.columns.get_level_values(0).unique():
        obs_vec = observation_vectors[obs_key]
        data_key = obs_vec.getDataKey()

        if obs_key not in summary_obs_keys:
            response_observation_link[data_key] = obs_key
        else:
            response_observation_link[data_key] = data_key
    return response_observation_link


def create_response_records(ert, ensemble_name: str, observations: List[dict]):
    data = {
        key.split("@")[0]: ert.gather_gen_data_data(case=ensemble_name, key=key)
        for key in ert.get_gen_data_keys()
    }

    data.update(
        {
            key: ert.gather_summary_data(case=ensemble_name, key=key)
            for key in ert.get_summary_keys()
        }
    )
    response_observation_links = _create_response_observation_links(ert)
    observation_ids = {obs["name"]: obs["id"] for obs in observations}
    records = []
    for key, response in data.items():
        realizations = {}
        for index, values in response.items():
            df = pd.DataFrame(values.to_list())
            df = df.transpose()
            df.columns = _prepare_x_axis(response.index.tolist())
            realizations[index] = df
        observation_key = response_observation_links.get(key)
        linked_observation = (
            [observation_ids[observation_key]] if observation_key else None
        )
        records.append(
            dict(name=key, data=realizations, observations=linked_observation)
        )
    return records


def _prepare_x_axis(x_axis: List[Union[int, float, str, pd.Timestamp]]) -> List[str]:
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]


def _get_obs_data(key, obs) -> dict:
    return dict(
        name=key,
        x_axis=obs.columns.get_level_values(0).to_list(),
        values=obs.loc["OBS"].to_list(),
        errors=obs.loc["STD"].to_list(),
    )


def create_observations(ert) -> List[Mapping[str, dict]]:
    observation_vectors = ert.get_observations()
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    if keys == []:
        return []

    data = MeasuredData(ert, keys, load_data=False)
    observations = data.data.loc[["OBS", "STD"]]
    grouped_obs = {}
    response_observation_link = {}

    for obs_key in observations.columns.get_level_values(0).unique():
        obs_vec = observation_vectors[obs_key]
        data_key = obs_vec.getDataKey()
        obs_data = _get_obs_data(obs_key, observations[obs_key])

        if obs_key not in summary_obs_keys:
            grouped_obs[obs_key] = obs_data
            response_observation_link[data_key] = obs_key
        else:
            response_observation_link[data_key] = data_key
            if data_key in grouped_obs:
                for el in filter(lambda x: not x == "name", obs_data):
                    grouped_obs[data_key][el] += obs_data[el]
            else:
                obs_data["name"] = data_key
                grouped_obs[data_key] = obs_data
    for key, obs in grouped_obs.items():
        x_axis, values, error = (
            list(t)
            for t in zip(*sorted(zip(obs["x_axis"], obs["values"], obs["errors"])))
        )
        x_axis = _prepare_x_axis(x_axis)
        grouped_obs[key]["x_axis"] = x_axis
        grouped_obs[key]["values"] = values
        grouped_obs[key]["errors"] = error
    return list(grouped_obs.values())


def _get_status(status):
    return status == "ACTIVE"


def _extract_active_observations(ert) -> Mapping[str, list]:
    update_step = ert.update_snapshots
    if not update_step:
        return {}
    # We make way to many assumptions here
    try:
        last_snapshot = list(update_step.keys())[-1]
        update_steps = update_step[last_snapshot].update_step_snapshots
        global_update_step = update_steps["ALL_ACTIVE"]
    except KeyError:
        logger.error(
            f"Failed to load global update_step snapshot, probably using "
            f"update_steps, expected only ALL_ACTIVE, found: {update_steps.keys()}"
        )
        return {}
    result = defaultdict(list)
    for obs_name, status in zip(
        global_update_step.obs_name, global_update_step.obs_status
    ):
        result[obs_name].append(_get_status(status))
    return result


def _create_observation_transformation(ert, db_observations) -> List[dict]:
    observation_vectors = ert.get_observations()
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    active_obs = _extract_active_observations(ert)
    transformations: Dict = {}
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    data = MeasuredData(ert, keys, load_data=False)
    observations = data.data.loc[["OBS", "STD"]]

    for obs_key, active_mask in active_obs.items():
        obs_data = _get_obs_data(obs_key, observations[obs_key])
        if obs_key in summary_obs_keys:
            obs_vec = observation_vectors[obs_key]
            data_key = obs_vec.getDataKey()
            if data_key in transformations:
                transformations[data_key]["x_axis"] += obs_data["x_axis"]
                transformations[data_key]["active"] += active_mask
                transformations[data_key]["scale"] += [1 for _ in active_mask]
            else:
                transformations[data_key] = dict(
                    name=data_key,
                    x_axis=obs_data["x_axis"],
                    scale=[1 for _ in active_mask],
                    active=active_mask,
                )
        else:
            # Scale is now mocked to 1 for now
            transformations[obs_key] = dict(
                name=obs_key,
                x_axis=obs_data["x_axis"],
                scale=[1 for _ in active_mask],
                active=active_mask,
            )
    observation_ids = {obs["name"]: obs["id"] for obs in db_observations}
    # Sorting by x_axis matches the transformation with the observation, mostly
    # needed for grouped summary obs
    for key, obs in transformations.items():
        x_axis, active, scale = (
            list(t)
            for t in zip(*sorted(zip(obs["x_axis"], obs["active"], obs["scale"])))
        )
        x_axis = _prepare_x_axis(x_axis)
        transformations[key]["x_axis"] = x_axis
        transformations[key]["active"] = active
        transformations[key]["scale"] = scale
        transformations[key]["observation_id"] = observation_ids[key]

    return [transformation for _, transformation in transformations.items()]


_PRIOR_NAME_MAP = {
    "NORMAL": "normal",
    "LOGNORMAL": "lognormal",
    "TRIANGULAR": "trig",
    "TRUNCATED_NORMAL": "ert_truncnormal",
    "CONST": "const",
    "UNIFORM": "uniform",
    "LOGUNIF": "loguniform",
    "DUNIF": "ert_duniform",
    "RAW": "stdnormal",
    "ERRF": "ert_erf",
    "DERRF": "ert_derf",
}


def create_priors(ert) -> Mapping[str, dict]:
    priors = {}
    for group, gen_kw_priors in ert.gen_kw_priors().items():
        for gen_kw_prior in gen_kw_priors:
            prior = {
                "function": _PRIOR_NAME_MAP[gen_kw_prior["function"]],
            }
            for arg_name, arg_value in gen_kw_prior["parameters"].items():
                # triangular uses X<arg_name>, removing the x prefix
                if arg_name.startswith("X"):
                    arg_name = arg_name[1:]
                # Libres calls it steps, but normal stats uses bins
                if arg_name == "STEPS":
                    arg_name = "bins"
                prior[arg_name.lower()] = arg_value

            priors[f"{group}:{gen_kw_prior['key']}"] = prior
    return priors


def _get_from_server(url, headers=None, status_code=200) -> requests.Response:
    if headers is None:
        headers = {}
    session = Storage.session()
    resp = session.get(
        url,
        headers=headers,
    )

    if resp.status_code != status_code:
        logger.error(f"Failed to fetch from {url}. Response: {resp.text}")
    return resp


def _post_to_server(
    url, data=None, params=None, json=None, headers=None, status_code=200
) -> requests.Response:
    if headers is None:
        headers = {}
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


@feature_enabled("new-storage")
def post_update_data(
    ert: "LibresFacade", parent_ensemble_id: str, algorithm: str
) -> str:
    observations = _get_from_server(
        f"ensembles/{parent_ensemble_id}/observations",
    ).json()

    # create update thingy
    update_create = dict(
        observation_transformations=_create_observation_transformation(
            ert, observations
        ),
        ensemble_reference_id=parent_ensemble_id,
        ensemble_result_id=None,
        algorithm=algorithm,
    )

    response = _post_to_server(
        "updates",
        json=update_create,
    )
    update = response.json()
    return update["id"]


@feature_enabled("new-storage")
def post_ensemble_results(
    ert: "LibresFacade", case_name: str, ensemble_id: str
) -> None:
    observations = _get_from_server(
        f"ensembles/{ensemble_id}/observations",
    ).json()

    for record in create_response_records(ert, case_name, observations):
        realizations = record["data"]
        name = record["name"]
        for index, data in realizations.items():
            stream = io.BytesIO()
            data.to_parquet(stream)
            _post_to_server(
                f"ensembles/{ensemble_id}/records/{name}/matrix",
                params={"realization_index": index, "record_class": "response"},
                data=stream.getvalue(),
                headers={"content-type": "application/x-parquet"},
            )
            if record["observations"] is not None:
                _post_to_server(
                    f"ensembles/{ensemble_id}/records/{name}/observations",
                    params={"realization_index": index},
                    json=record["observations"],
                )


@feature_enabled("new-storage")
def post_ensemble_data(
    ert: "LibresFacade",
    case_name: str,
    ensemble_size: int,
    update_id: Optional[str] = None,
    active_realizations: Optional[List[int]] = None,
) -> str:
    if active_realizations is None:
        active_realizations = []

    if update_id is None:
        exp_response = _post_to_server(
            "experiments",
            json=create_experiment(ert),
        ).json()
        experiment_id = exp_response["id"]
        for obs in create_observations(ert):
            _post_to_server(
                f"experiments/{experiment_id}/observations",
                json=obs,
            )
    else:
        update = _get_from_server(f"updates/{update_id}").json()
        experiment_id = update["experiment_id"]

    parameters = create_parameters(ert)
    response_names = [
        key.split("@")[0] if ert.is_gen_data_key(key) else key
        for key in ert.all_data_type_keys()
        if ert.is_gen_data_key(key) or ert.is_summary_key(key)
    ]

    ens_response = _post_to_server(
        f"experiments/{experiment_id}/ensembles",
        json=create_ensemble(
            size=ensemble_size,
            case_name=case_name,
            parameter_names=[param["name"] for param in parameters],
            response_names=response_names,
            update_id=update_id,
            active_realizations=active_realizations,
        ),
    )

    ensemble_id = ens_response.json()["id"]

    for param in parameters:
        df = pd.DataFrame([p.tolist() for p in param["values"]])
        _post_to_server(
            f"ensembles/{ensemble_id}/records/{param['name']}/matrix",
            data=df.to_csv(),
            headers={"content-type": "text/csv"},
        )

    return ensemble_id
