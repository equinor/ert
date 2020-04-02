import flask
import werkzeug.exceptions as werkzeug_exc
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi
from ert_shared.storage import connections
from flask import Response, request


def resolve_ensemble_uri(ensemble_ref):
    BASE_URL = request.host_url
    return "{}ensembles/{}".format(BASE_URL, ensemble_ref)


def resolve_ref_uri(struct, ensemble_id=None):
    if isinstance(struct, list):
        for item in struct:
            resolve_ref_uri(item, ensemble_id)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            split_key = key.split("_")

            if len(split_key) == 2 and split_key[1] == "ref" in key:
                type_name = split_key[0]
                if type_name == "realization":
                    base = resolve_ensemble_uri(ensemble_id)
                    struct["ref_url"] = "{}/realizations/{}".format(base, val)
                elif type_name == "ensemble":
                    struct["ref_url"] = resolve_ensemble_uri(val)
                elif type_name == "response":
                    base = resolve_ensemble_uri(ensemble_id)
                    struct["ref_url"] = "{}/responses/{}".format(base, val)
                elif type_name == "data":
                    struct["data_url"] = "{}data/{}".format(request.host_url, val)
                else:
                    continue
                del struct[key]
            else:
                resolve_ref_uri(val, ensemble_id)


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
        self.app.add_url_rule(
            "/observation/<name>",
            "get_observation",
            self.get_observation,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/observation/<name>/attributes",
            "get_observation_attributes",
            self.get_observation_attributes,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/observation/<name>/attributes",
            "set_observation_attributes",
            self.set_observation_attributes,
            methods=["POST"],
        )

    def ensembles(self):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ensembles = api.ensembles()
            resolve_ref_uri(ensembles)
            return ensembles

    def ensemble_by_id(self, ensemble_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ensemble = api.ensemble_schema(ensemble_id)
            resolve_ref_uri(ensemble, ensemble_id)
            return ensemble

    def realizations(self, ensemble_id):
        pass

    def realization_by_id(self, ensemble_id, realization_idx):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            realization = api.realization(ensemble_id, realization_idx, None)
            resolve_ref_uri(realization, ensemble_id)
            return realization

    def response_by_name(self, ensemble_id, response_name):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            response = api.response(ensemble_id, response_name, None)
            resolve_ref_uri(response, ensemble_id)
            return response

    def data(self, data_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            data = api.data(data_id)
            if isinstance(data, list):
                return ",".join([str(x) for x in data])
            else:
                return str(data)

    def get_observation(self, name):
        """Return an observation."""
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            obs = api.observation(name)
            if obs is None:
                raise werkzeug_exc.NotFound()
            return obs

    def get_observation_attributes(self, name):
        """Return attributes for an observation.

        {
            "attributes": {...}
        }
        """
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            attrs = api.get_observation_attributes(name)
            if attrs is None:
                raise werkzeug_exc.NotFound()
            return attrs

    def set_observation_attributes(self, name):
        """Set attributes on an observation.

        The posted JSON will be expected to be
        {
            "attributes": {
                "region": "1",
                "depth": "2892.1"
            }
        }
        """
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            js = request.get_json()
            for k, v in js["attributes"].items():
                obs = api.set_observation_attribute(name, k, v)
                if obs is None:
                    raise werkzeug_exc.NotFound()
            return api.observation(name), 201


def run_server(args):
    wrapper = FlaskWrapper(
        rdb_url="sqlite:///entities.db", blob_url="sqlite:///blobs.db"
    )
    wrapper.app.run(host="0.0.0.0", debug=True)
