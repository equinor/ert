from typing import List, Optional, Mapping, Tuple

from ert_data.measured import MeasuredData
from res.enkf.enums.enkf_obs_impl_type_enum import EnkfObservationImplementationType
from ert_shared.ert_adapter import ERT, LibresFacade
from ert_shared.feature_toggling import feature_enabled
from ert_shared.storage.server_monitor import ServerMonitor
from res.enkf.export import MisfitCollector

import ert_shared.storage.json_schema as js
import requests


class _EnKFMain:
    """MisfitCollector.createActiveList, used by create_update_data, expects an
    EnKFMain object as its first parameter. However, this function only uses the
    getEnsembleSize function of EnKFMain. This class is a mock class that provides
    only this one function.
    """

    def __init__(self, ert: LibresFacade):
        self._ensemble_size = ert.get_ensemble_size()

    def getEnsembleSize(self) -> int:
        return self._ensemble_size


def create_ensemble(ert, update_id: int = None) -> js.EnsembleCreate:
    ensemble_name = ert.get_current_case_name()

    priors = []

    priors = [
        js.PriorCreate(
            group=group,
            key=prior["key"],
            function=prior["function"],
            parameter_names=list(prior["parameters"].keys()),
            parameter_values=list(prior["parameters"].values()),
        )
        for group, priors in ert.gen_kw_priors().items()
        for prior in priors
    ]

    parameters = [
        js.ParameterCreate(
            group=key[: key.index(":")],
            name=key[key.index(":") + 1 :],
            values=list(parameter.values),
        )
        for key, parameter in (
            (key, ert.gather_gen_kw_data(ensemble_name, key))
            for key in ert.all_data_type_keys()
            if ert.is_gen_kw_key(key)
        )
    ]

    observations, response_observation_link = create_observations(ert)

    return js.EnsembleCreate(
        name=ensemble_name,
        realizations=ert.get_ensemble_size(),
        priors=priors,
        parameters=parameters,
        observations=observations,
        response_observation_link=response_observation_link,
        update_id=update_id,
    )


def create_responses(ert, ensemble_name: str) -> List[js.ResponseCreate]:
    data = {
        key.split("@")[0]: ert.gather_gen_data_data(case=ensemble_name, key=key)
        for key in ert.all_data_type_keys()
        if ert.is_gen_data_key(key)
    }

    data.update(
        {
            key: ert.gather_summary_data(case=ensemble_name, key=key)
            for key in ert.all_data_type_keys()
            if ert.is_summary_key(key)
        }
    )

    return [
        js.ResponseCreate(
            name=key,
            indices=response.index.to_list(),
            realizations={
                index: values.to_list() for index, values in response.iteritems()
            },
        )
        for key, response in data.items()
    ]


def _get_obs_data(key, obs):
    return dict(
        name=key,
        x_axis=obs.columns.get_level_values(0).to_list(),
        values=obs.loc["OBS"].to_list(),
        errors=obs.loc["STD"].to_list(),
    )


def create_observations(ert) -> Tuple[List[js.ObservationCreate], Mapping[str, str]]:
    observation_vectors = ert.get_observations()
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    if keys == []:
        return [], {}

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
        grouped_obs[key]["x_axis"] = x_axis
        grouped_obs[key]["values"] = values
        grouped_obs[key]["errors"] = error
    return [
        js.ObservationCreate(**obs) for obs in grouped_obs.values()
    ], response_observation_link


def _extract_active_observations(ert):
    update_step = ert.get_update_step()
    if len(update_step) == 0:
        return {}

    ministep = update_step[-1]
    obs_data = ministep.get_obs_data()
    if obs_data is None:
        return {}

    active_obs = {}
    for block_num in range(obs_data.get_num_blocks()):
        block = obs_data.get_block(block_num)
        obs_key = block.get_obs_key()
        active_list = [block.is_active(i) for i in range(len(block))]
        active_obs[obs_key] = active_list
    return active_obs


def _create_observation_transformation(ert):
    observation_vectors = ert.get_observations()
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    active_obs = _extract_active_observations(ert)
    transformations = {}
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

    # Sorting by x_axis matches the transformation with the observation, mostly needed for grouped summary obs
    for key, obs in transformations.items():
        x_axis, active, scale = (
            list(t)
            for t in zip(*sorted(zip(obs["x_axis"], obs["active"], obs["scale"])))
        )
        transformations[key]["x_axis"] = x_axis
        transformations[key]["active"] = active
        transformations[key]["scale"] = scale

    return [
        js.ObservationTransformationCreate(**transformation)
        for _, transformation in transformations.items()
    ]


@feature_enabled("new-storage")
def post_update_data(parent_ensemble_id: int, algorithm: str) -> int:
    server = ServerMonitor.get_instance()
    ert = ERT.enkf_facade

    # create update thingy
    update_create = js.UpdateCreate(
        observation_transformations=_create_observation_transformation(ert),
        ensemble_reference_id=parent_ensemble_id,
        ensemble_result_id=None,
        algorithm=algorithm,
    )

    # URL should not be to ensembles
    response = requests.post(
        f"{server.fetch_url()}/ensembles/{parent_ensemble_id}/updates",
        data=update_create.json(),
        auth=server.fetch_auth(),
    )
    update = js.Update.parse_obj(response.json())
    return update.id


@feature_enabled("new-storage")
def post_ensemble_results(ensemble_id: int):
    server = ServerMonitor.get_instance()
    ert = ERT.enkf_facade

    for r in create_responses(ert, ert.get_current_case_name()):
        requests.post(
            f"{server.fetch_url()}/ensembles/{ensemble_id}/responses",
            data=r.json(),
            auth=server.fetch_auth(),
        )


@feature_enabled("new-storage")
def post_ensemble_data(update_id: int = None) -> int:
    server = ServerMonitor.get_instance()
    ert = ERT.enkf_facade
    response = requests.post(
        f"{server.fetch_url()}/ensembles",
        data=create_ensemble(ert, update_id).json(),
        auth=server.fetch_auth(),
    )
    if not response.status_code == 200:
        raise RuntimeError(response.text)

    ens = js.Ensemble.parse_obj(response.json())

    return ens.id
