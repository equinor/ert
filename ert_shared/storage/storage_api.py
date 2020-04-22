from io import StringIO

import pandas as pd
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage import connections


class StorageApi(object):
    def __init__(self, rdb_url, blob_url):
        self._rdb_connection = connections.get_rdb_connection(rdb_url)
        self._rdb_api = RdbApi(connection=self._rdb_connection)

        self._blob_connection = connections.get_blob_connection(blob_url)
        self._blob_api = BlobApi(connection=self._blob_connection)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._rdb_connection.close()
        self._blob_connection.close()

    def _ensemble_minimal(self, ensemble):
        return {
            "name": ensemble.name,
            "time_created": ensemble.time_created.isoformat(),
            "ensemble_ref": ensemble.id,
            "parent": {
                "ensemble_ref": ensemble.parent.ensemble_reference.id,
                "name": ensemble.parent.ensemble_reference.name,
            }
            if ensemble.parent is not None
            else None,
            "children": [
                {
                    "ensemble_ref": child.ensemble_result.id,
                    "name": child.ensemble_result.name,
                }
                for child in ensemble.children
            ],
        }

    def ensembles(self, filter=None):
        with self._rdb_api as rdb_api:
            data = [
                self._ensemble_minimal(ensemble)
                for ensemble in rdb_api.get_all_ensembles()
            ]

        return {"ensembles": data}

    def realization(self, ensemble_id, realization_idx, filter):
        with self._rdb_api as rdb_api:
            realization = (
                rdb_api.get_realizations_by_ensemble_id(ensemble_id=ensemble_id)
                .filter_by(index=realization_idx)
                .one()
            )
            response_definitions = rdb_api.get_response_definitions_by_ensemble_id(
                ensemble_id=ensemble_id
            )
            responses = [
                {
                    "name": resp_def.name,
                    "response": rdb_api.get_response_by_realization_id(
                        response_definition_id=resp_def.id,
                        realization_id=realization.id,
                    ),
                }
                for resp_def in response_definitions
            ]

            parameter_definitions = rdb_api.get_parameter_definitions_by_ensemble_id(
                ensemble_id=ensemble_id
            )
            parameters = [
                {
                    "name": param_def.name,
                    "parameter": rdb_api.get_parameter_by_realization_id(
                        parameter_definition_id=param_def.id,
                        realization_id=realization.id,
                    ),
                }
                for param_def in parameter_definitions
            ]

        return_schema = {
            "name": realization_idx,
            "responses": [
                {"name": res["name"], "data_ref": res["response"].values_ref}
                for res in responses
            ],
            "parameters": [
                {"name": par["name"], "data_ref": par["parameter"].value_ref}
                for par in parameters
            ],
        }

        return return_schema

    def _calculate_misfit(
        self, obs_value, response_values, obs_stds, obs_data_indexes, obs_index
    ):
        observation_std = obs_stds[obs_index]
        response_index = obs_data_indexes[obs_index]
        response_value = response_values[response_index]
        difference = response_value - obs_value
        misfit = (difference / observation_std) ** 2
        sign = difference > 0

        return {"value": misfit, "sign": sign, "obs_index": obs_index}

    def response(self, ensemble_id, response_name, filter):
        with self._rdb_api as rdb_api, self._blob_api as blob_api:
            bundle = rdb_api.get_response_bundle(
                response_name=response_name, ensemble_id=ensemble_id
            )

            observation_links = bundle.observation_links
            responses = bundle.responses
            univariate_misfits = {}
            for resp in responses:
                resp_values = list(blob_api.get_blob(resp.values_ref).data)
                univariate_misfits[resp.realization.index] = {}
                for link in observation_links:
                    observation = link.observation
                    obs_values = list(blob_api.get_blob(observation.values_ref).data)
                    obs_stds = list(blob_api.get_blob(observation.stds_ref).data)
                    obs_data_indexes = list(
                        blob_api.get_blob(observation.data_indexes_ref).data
                    )
                    misfits = []
                    for obs_index, obs_value in enumerate(obs_values):
                        misfits.append(
                            self._calculate_misfit(
                                obs_value,
                                resp_values,
                                obs_stds,
                                obs_data_indexes,
                                obs_index,
                            )
                        )
                    univariate_misfits[resp.realization.index][
                        observation.name
                    ] = misfits

            return_schema = {
                "name": response_name,
                "realizations": [
                    {
                        "name": resp.realization.index,
                        "realization_ref": resp.realization.index,
                        "data_ref": resp.values_ref,
                        "summarized_misfits": {
                            misfit.observation_response_definition_link.observation.name: misfit.value
                            for misfit in resp.misfits
                        },
                        "univariate_misfits": {
                            obs_name: misfits
                            for obs_name, misfits in univariate_misfits[
                                resp.realization.index
                            ].items()
                        },
                    }
                    for resp in responses
                ],
                "axis": {"data_ref": bundle.indexes_ref},
            }
            if len(observation_links) > 0:
                return_schema["observations"] = [
                    self._obs_to_json(link.observation, link.active_ref)
                    for link in observation_links
                ]

        return return_schema

    def data(self, id):
        with self._blob_api as blob_api:
            return_data = blob_api.get_blob(id).data
        return return_data

    def observation(self, name):
        with self._rdb_api as rdb_api:
            obs = rdb_api.get_observation(name)
            return None if obs is None else self._obs_to_json(obs)

    def get_observation_attributes(self, name):
        with self._rdb_api as rdb_api:
            attrs = rdb_api.get_observation_attributes(name)
            return None if attrs is None else {"attributes": attrs}

    def get_observation_attribute(self, name, attribute):
        with self._rdb_api as rdb_api:
            attr = rdb_api.get_observation_attribute(name, attribute)
            return None if attr is None else {"attributes": {attribute: attr}}

    def set_observation_attribute(self, name, attribute, value):
        with self._rdb_api as rdb_api:
            obs = rdb_api.add_observation_attribute(name, attribute, value)
            if obs is None:
                return None
            rdb_api.commit()
            return self._obs_to_json(obs)

    def ensemble_schema(self, ensemble_id):
        with self._rdb_api as rdb_api:
            ens = rdb_api.get_ensemble_by_id(ensemble_id)
            return_schema = self._ensemble_minimal(ens)
            return_schema.update(
                {
                    "realizations": [
                        {"name": real.index, "realization_ref": real.index}
                        for real in rdb_api.get_realizations_by_ensemble_id(ensemble_id)
                    ],
                    "responses": [
                        {"name": resp.name, "response_ref": resp.name}
                        for resp in rdb_api.get_response_definitions_by_ensemble_id(
                            ensemble_id
                        )
                    ],
                    "parameters": [
                        self._parameter_minimal(
                            name=par.name,
                            group=par.group,
                            prior=par.prior,
                            parameter_def_id=par.id,
                        )
                        for par in rdb_api.get_parameter_definitions_by_ensemble_id(
                            ensemble_id
                        )
                    ],
                }
            )
        return return_schema

    def _obs_to_json(self, obs, active_ref=None):
        data = {
            "name": obs.name,
            "data": {
                "values": {"data_ref": obs.values_ref},
                "std": {"data_ref": obs.stds_ref},
                "data_indexes": {"data_ref": obs.data_indexes_ref},
                "key_indexes": {"data_ref": obs.key_indexes_ref},
            },
        }
        if active_ref is not None:
            data["data"]["active_mask"] = {"data_ref": active_ref}

        attrs = obs.get_attributes()
        if len(attrs) > 0:
            data["attributes"] = attrs

        return data

    def _parameter_minimal(self, name, group, prior, parameter_def_id):
        return {
            "key": name,
            "group": group,
            "parameter_ref": parameter_def_id,
            "prior": {
                "function": prior.function,
                "parameter_names": prior.parameter_names,
                "parameter_values": prior.parameter_values,
            }
            if prior is not None
            else {},
        }

    def parameter(self, ensemble_id, parameter_def_id):
        with self._rdb_api as rdb_api:
            bundle = rdb_api.get_parameter_bundle(
                parameter_def_id=parameter_def_id, ensemble_id=ensemble_id
            )

            return_schema = self._parameter_minimal(
                name=bundle.name,
                group=bundle.group,
                prior=bundle.prior,
                parameter_def_id=parameter_def_id,
            )

            return_schema["parameter_realizations"] = [
                {
                    "name": param.realization.index,
                    "data_ref": param.value_ref,
                    "realization": {"realization_ref": param.realization.index},
                }
                for param in bundle.parameters
            ]
            return return_schema
