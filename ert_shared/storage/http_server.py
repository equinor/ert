import flask
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi
from flask import Response, request


def resolve_ensemble_uri(ensemble_ref):
    BASE_URL = request.host_url
    return "{}ensembles/{}".format(BASE_URL, ensemble_ref)


def resolve_data_uri(struct):
    BASE_URL = request.host_url
    if isinstance(struct, list):
        for item in struct:
            resolve_data_uri(item)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            if key == "data_ref":
                url = "{}data/{}".format(BASE_URL, val)
                struct["data_url"] = url
            else:
                resolve_data_uri(val)


def resolve_ref_uri(BASE_URL, struct):
    if isinstance(struct, list):
        for item in struct:
            resolve_ref_uri(BASE_URL, item)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            if "_ref" in key and "data" not in key:
                url = "{}/{}".format(BASE_URL, val)
                struct["ref_url"] = url
            else:
                resolve_ref_uri("{}/{}".format(BASE_URL, key), val)


class FlaskWrapper:
    def __init__(self, rdb_api=None, blob_api=None):
        self._rdb_api = rdb_api
        self._blob_api = blob_api

        self.app = flask.Flask("ert http api")
        self.app.add_url_rule("/ensembles", "ensembles", self.ensembles)
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>", "ensemble", self.ensemble_by_id
        )
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>/realizations/<realization_idx>",
            "realization",
            self.realization_by_id,
        )
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>/responses/<response_name>",
            "response",
            self.response_by_name,
        )
        self.app.add_url_rule("/data/<int:data_id>", "data", self.data)

    def ensembles(self):
        api = StorageApi(rdb_api=self._rdb_api, blob_api=self._blob_api)
        ensembles = api.ensembles()
        resolve_ref_uri("{}ensembles".format(request.host_url), ensembles)
        return {"ensembles": ensembles}

    def ensemble_by_id(self, ensemble_id):
        api = StorageApi(rdb_api=self._rdb_api, blob_api=self._blob_api)
        ensemble = api.ensemble_schema(ensemble_id)
        base_url = resolve_ensemble_uri(ensemble_id)
        resolve_ref_uri(base_url, ensemble)
        return ensemble

    def realizations(self, ensemble_id):
        pass

    def realization_by_id(self, ensemble_id, realization_idx):
        api = StorageApi(rdb_api=self._rdb_api, blob_api=self._blob_api)
        realization = api.realization(ensemble_id, realization_idx, None)
        resolve_data_uri(realization)
        return realization

    def response_by_name(self, ensemble_id, response_name):
        api = StorageApi(rdb_api=self._rdb_api, blob_api=self._blob_api)
        response = api.response(ensemble_id, response_name, None)
        base_url = resolve_ensemble_uri(ensemble_id)
        resolve_ref_uri(base_url, response)
        resolve_data_uri(response)
        return response

    def data(self, data_id):
        api = StorageApi(rdb_api=self._rdb_api, blob_api=self._blob_api)
        data = api.data(data_id)
        if isinstance(data, list):
            return ",".join([str(x) for x in data])
        else:
            return str(data)


def run_server(args):
    wrapper = FlaskWrapper(rdb_api=None, blob_api=None)
    wrapper.app.run()
