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


def create_ensemble(ert, reference: Optional[Tuple[int, str]]) -> js.EnsembleCreate:
    ensemble_name = ert.get_current_case_name()

    priors = []
    update = None

    if reference is None:
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
    else:
        update = js.UpdateCreate(ensemble_id=reference[0], algorithm=reference[1])

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
        update=update,
        observations=observations,
        response_observation_link=response_observation_link,
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


def create_observations(ert) -> Tuple[List[js.ObservationCreate], Mapping[str, str]]:
    observation_vectors = ert.get_observations()
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )

    def _get_obs_data(key, obs):
        return dict(
            name=key,
            x_axis=obs.columns.get_level_values(0).to_list(),
            values=obs.loc["OBS"].to_list(),
            errors=obs.loc["STD"].to_list(),
        )

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


def create_update_data(ert):
    fs = ert.get_current_fs()
    realizations = MisfitCollector.createActiveList(_EnKFMain(ert), fs)

    active_obs = _extract_active_observations(ert)

    for obs_vector in ert.get_observations():
        obs_key = obs_vector.getObservationKey()
        resp_key = obs_vector.getDataKey()
        active_blob = active_obs[obs_key] if active_obs else None

        yield js.MisfitCreate(
            observation_key=obs_key,
            response_definition_key=resp_key,
            active=active_blob,
            realizations={
                index: obs_vector.getTotalChi2(fs, index) for index in realizations
            },
        )


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

    for u in create_update_data(ert):
        requests.post(
            f"{server.fetch_url()}/ensembles/{ensemble_id}/misfit",
            data=u.json(),
            auth=server.fetch_auth(),
        )


@feature_enabled("new-storage")
def post_ensemble_data(reference: Optional[Tuple[int, str]] = None) -> int:
    server = ServerMonitor.get_instance()
    ert = ERT.enkf_facade
    response = requests.post(
        f"{server.fetch_url()}/ensembles",
        data=create_ensemble(ert, reference).json(),
        auth=server.fetch_auth(),
    )
    if not response.status_code == 200:
        raise RuntimeError(response.text)

    ens = js.Ensemble.parse_obj(response.json())

    return ens.id
