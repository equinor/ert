import flask
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi
from ert_shared.storage import connections
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
                del struct[key]
            else:
                resolve_data_uri(val)


def resolve_ref_uri(BASE_URL, struct, ensemble_id=None):
    if isinstance(struct, list):
        for item in struct:
            resolve_ref_uri(BASE_URL, item, ensemble_id)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            split_key = key.split("_")

            if len(split_key) == 2 and split_key[1] == "ref" in key and "data" not in key:
                type_name = split_key[0]
                if type_name == "realization":
                    base = resolve_ensemble_uri(ensemble_id)
                    url = "{}/realizations/{}".format(base, val)
                elif type_name == "ensemble":
                    url = resolve_ensemble_uri(val)
                elif type_name == "response":
                    base = resolve_ensemble_uri(ensemble_id)
                    url = "{}/responses/{}".format(base, val)
                else:
                    url = key
                struct["ref_url"] = url
                del struct[key]
            else:
                resolve_ref_uri("{}/{}".format(BASE_URL, key), val, ensemble_id)


class FlaskWrapper:
    def __init__(self, rdb_url=None, blob_url=None):
        self._rdb_url = rdb_url
        self._blob_url = blob_url

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
        with StorageApi(
            rdb_url=self._rdb_url, blob_url=self._blob_url
        ) as api:
            ensembles = api.ensembles()
            resolve_ref_uri("{}ensembles".format(request.host_url), ensembles)
            return {"ensembles": ensembles}

    def ensemble_by_id(self, ensemble_id):
        with StorageApi(
            rdb_url=self._rdb_url, blob_url=self._blob_url
        ) as api:
            ensemble = api.ensemble_schema(ensemble_id)
            base_url = resolve_ensemble_uri(ensemble_id)
            resolve_ref_uri(base_url, ensemble)
            return ensemble

    def realizations(self, ensemble_id):
        pass

    def realization_by_id(self, ensemble_id, realization_idx):
        with StorageApi(
            rdb_url=self._rdb_url, blob_url=self._blob_url
        ) as api:
            realization = api.realization(ensemble_id, realization_idx, None)
            resolve_data_uri(realization)
            return realization

    def response_by_name(self, ensemble_id, response_name):
        with StorageApi(
            rdb_url=self._rdb_url, blob_url=self._blob_url
        ) as api:
            response = api.response(ensemble_id, response_name, None)
            base_url = resolve_ensemble_uri(ensemble_id)
            resolve_ref_uri(base_url, response)
            resolve_data_uri(response)
            return response

    def data(self, data_id):
        with StorageApi(
            rdb_url=self._rdb_url, blob_url=self._blob_url
        ) as api:
            data = api.data(data_id)
            if isinstance(data, list):
                return ",".join([str(x) for x in data])
            else:
                return str(data)


def run_server(args):
    wrapper = FlaskWrapper(rdb_url="sqlite:///entities.db", blob_url="sqlite:///blobs.db")
    wrapper.app.run(host="0.0.0.0")
