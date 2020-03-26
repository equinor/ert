from io import StringIO

import pandas as pd
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.blob_api import BlobApi


class StorageApi(object):

    def __init__(self, session=None, blob_session=None):
        self._session = session
        self._blob_session = blob_session

    def _repo(self):
        if self._session is not None:
            return RdbApi(self._session)
        else:
            return RdbApi()

    def _blob(self):
        if self._session is not None:
            return BlobApi(self._session)
        else:
            return BlobApi()

    def ensembles(self, filter=None):
        """
        This function returns an overview of the ensembles available in the database
        @return_type: 
        [
            {
                "name" : "default"
                "ref_pointer" : "10" -> "/ensembles/10" 
            }
        ]
        """
        return [{"name": ensemble.name, "ref_pointer" : ensemble.name} for ensemble in self._repo().get_all_ensembles()]


    def realizations(self, ensemble_id, filter): 
        """
        This function returns an overview of the realizations in a given ensemble
        @return_type: 
        [
            {
                "name" : "<iens>"
                "ref_pointer" : "2" -> "/ensembles/<ensemble_id>/realizations/2"
                "data_pointer" : "<key>" -> "/data/<key>" 
            }
        ]
        """
        pass
    
    def data(self, id):
        return self._blob().get_blob(id)


    def ensemble_schema(self, ensemble):
        repo = self._repo()
        ens = repo.get_ensemble(ensemble)

        schema = {
            "name": ens.name,
            "parameters": [],
            "responses": [],
            "observations": []
        }

        for param_def in ens.parameter_definitions:
            reals = []
            schema["parameters"].append({
                "name": param_def.name,
                "realizations": reals
            })

            for real in ens.realizations:
                for param in real.parameters:
                    if param.parameter_definition == param_def:
                        reals.append({"name": real.index,
                                      "data_refs": {"value": param.value_ref}})

        def observation_names(resp_def):
            if resp_def.observation is not None:
                return [resp_def.observation.name]
            else:
                return []

        for resp_def in ens.response_definitions:
            reals = []
            schema["responses"].append({
                "name": resp_def.name,
                "realizations": reals
            })

            for real in ens.realizations:
                for resp in real.responses:
                    if resp.response_definition == resp_def:
                        reals.append({"name": real.index,
                                      "data_refs": {"values": resp.values_ref},
                                      "observed_by": observation_names(resp_def)})

        for obs_name in repo.get_all_observation_keys():
            obs = repo.get_observation(obs_name)
            schema["observations"].append({
                "name":obs.name,
                "data_refs": {
                    "values": obs.values_ref,
                    "stds": obs.stds_ref,
                    "key_indexes": obs.key_indexes_ref,
                    "data_indexes": obs.data_indexes_ref,
                },
                "observes": [resp_def.name for resp_def in obs.response_definitions]
            })

        return schema
