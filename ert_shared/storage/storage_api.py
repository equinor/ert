from io import StringIO

import pandas as pd
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi


class StorageApi(object):
    def __init__(self, rdb_api=None, blob_api=None):
        if rdb_api is None:
            self._rdb_api = RdbApi()
        else:
            self._rdb_api = rdb_api

        if blob_api is None:
            self._blob_api = BlobApi()
        else:
            self._blob_api = blob_api

    def ensembles(self, filter=None):
        """
        This function returns an overview of the ensembles available in the database
        @return_type:
        [
            {
                "name" : "default"
                "ref_pointer" : "<ensemble_id>" -> "/ensembles/<ensemble_id>"
            }
        ]
        """
        with self._rdb_api as rdb_api:
            data = [
                {"name": ensemble.name, "ref_pointer": ensemble.id}
                for ensemble in rdb_api.get_all_ensembles()
            ]
        return data

    def realization(self, ensemble_id, realization_idx, filter):
        """
        This function returns an overview of the realizations in a given ensemble
        @return_type:
        {
            "name" : "<real.index>
            "ensemble_id" : <ensemble_id>
            "responses: [
                {
                    "name" : "<response key>"
                    "ref_pointer" : "<response key>" -> "/ensembles/<ensemble_id>/realizations/<realization_id>/responses/<response_key>"
                    "data_pointer" : "<key>" -> "/data/<key>"
                }
            ]
            "parameters" : [
                {
                    "name" : <parameter_name>
                    "data_pointer" : <parameter_values_ref> # maybe serve value directly?
                }
            ]
        }
        """
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
            "ensemble_id": ensemble_id,
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

    def response(self, ensemble_id, response_name, filter):
        """
        This function returns an overview of the response in a given ensemble
        @return_type: 
        {
            "name" : "<name>
            "ensemble_id" : <ensemble_id>
            "realizations" : [
                {
                    "name" : "<realization_idx>"
                    "ref_pointer" : "<realization_idx>" 
                    "data_pointer" : "<key>" -> "/data/<key>" 
                }
            ]
            "axis" : {
                "data_pointer": <indexes_ref>
            }
            "observation": {
                "data" : [
                    {
                        "name" : "values"
                        "data_pointer" : <values_ref>
                    }
                    {
                        "name" : "std"
                        "data_pointer" : <stds_ref>
                    }
                    {
                        "name" : "data_indexes"
                        "data_pointer" : <data_indexes_ref>
                    }
                ]
            }
        }
        """

        with self._rdb_api as rdb_api:
            bundle = self._rdb_api.get_response_bundle(
                response_name=response_name, ensemble_id=ensemble_id
            )

            observation = bundle.observation
            responses = bundle.responses

            return_schema = {
                "name": response_name,
                "ensemble_id": ensemble_id,
                "realizations": [
                    {
                        "name": resp.realization.index,
                        "ref_pointer": resp.realization.index,
                        "data_pointer": resp.values_ref,
                    }
                    for resp in responses
                ],
                "axis": {"data_pointer": bundle.indexes_ref},
            }
            if observation is not None:
                return_schema["observation"] = {
                    "data": [
                        {"name": "values", "data_pointer": observation.values_ref},
                        {"name": "std", "data_pointer": observation.stds_ref},
                        {
                            "name": "data_indexes",
                            "data_pointer": observation.data_indexes_ref,
                        },
                    ]
                }

        return return_schema

    def data(self, id):
        with self._blob_api as blob_api:
            return_data = blob_api.get_blob(id).data
        return return_data

    def ensemble_schema(self, ensemble_id):
        """
        @return_type:

        {
            "name" : "<ensemble name>"
            "realizations" : [
                {
                    "name" : "<realization_idx>"
                    "ref_pointer: "<realization_idx>" -> "/ensembles/<ensemble_id>/realizations/<realization_idx>"
                }
            ]
            "responses" : [
                {
                    "name" : "<name>"
                    "ref_pointer: "<name>" -> "/ensembles/<ensemble_id>/responses/<name>"
                }
            ]
            "parameters": [
                {
                    "name": "<name>"
                }
            ]
        }

        """

        with self._rdb_api as rdb_api:
            ens = rdb_api.get_ensemble_by_id(ensemble_id)
            return_schema = {
                "name": ens.name,
                "realizations": [
                    {"name": real.index, "ref_pointer": real.index}
                    for real in rdb_api.get_realizations_by_ensemble_id(ensemble_id)
                ],
                "responses": [
                    {"name": resp.name, "ref_pointer": resp.name}
                    for resp in rdb_api.get_response_definitions_by_ensemble_id(
                        ensemble_id
                    )
                ],
                "parameters": [
                    {"name": par.name}
                    for par in rdb_api.get_parameter_definitions_by_ensemble_id(
                        ensemble_id
                    )
                ],
            }

        return return_schema

        schema = {
            "name": ens.name,
            "parameters": [],
            "responses": [],
            "observations": [],
        }

        for param_def in ens.parameter_definitions:
            reals = []
            schema["parameters"].append({"name": param_def.name, "realizations": reals})

            for real in ens.realizations:
                for param in real.parameters:
                    if param.parameter_definition == param_def:
                        reals.append(
                            {
                                "name": real.index,
                                "data_refs": {"value": param.value_ref},
                            }
                        )

        def observation_names(resp_def):
            if resp_def.observation is not None:
                return [resp_def.observation.name]
            else:
                return []

        for resp_def in ens.response_definitions:
            reals = []
            schema["responses"].append({"name": resp_def.name, "realizations": reals})

            for real in ens.realizations:
                for resp in real.responses:
                    if resp.response_definition == resp_def:
                        reals.append(
                            {
                                "name": real.index,
                                "data_refs": {"values": resp.values_ref},
                                "observed_by": observation_names(resp_def),
                            }
                        )

        for obs_name in rdb_api.get_all_observation_keys():
            obs = rdb_api.get_observation(obs_name)
            schema["observations"].append(
                {
                    "name": obs.name,
                    "data_refs": {
                        "values": obs.values_ref,
                        "stds": obs.stds_ref,
                        "key_indexes": obs.key_indexes_ref,
                        "data_indexes": obs.data_indexes_ref,
                    },
                    "observes": [
                        resp_def.name for resp_def in obs.response_definitions
                    ],
                }
            )

        return schema
