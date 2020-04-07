import flask
import os
import yaml
import werkzeug.exceptions as werkzeug_exc
from ert_shared.storage.storage_api import StorageApi
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
                elif type_name == "parameter":
                    base = resolve_ensemble_uri(ensemble_id)
                    struct["ref_url"] = "{}/parameters/{}".format(base, val)
                elif type_name == "data":
                    struct["data_url"] = "{}data/{}".format(request.host_url, val)
                elif type_name == "alldata":
                    struct["alldata_url"] = "{}/data".format(request.url)
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
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>/responses/<response_name>/data",
            "response_data",
            self.response_data_by_name,
        )
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>/parameters/<parameter_def_id>",
            "parameter",
            self.parameter_by_id,
        )
        self.app.add_url_rule(
            "/ensembles/<ensemble_id>/parameters/<parameter_def_id>/data",
            "parameter_data",
            self.parameter_data_by_id,
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
        self.app.add_url_rule("/shutdown", "shutdown", self.shutdown, methods=["POST"])
        self.app.add_url_rule(
            "/schema.json", "schema", self.schema, methods=["GET"],
        )

    def schema(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(cur_path, "oas.yml")
        with open(schema_file) as f:
            return yaml.safe_load(f)

    def ensembles(self):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ensembles = api.get_ensembles()
            resolve_ref_uri(ensembles)
            return ensembles

    def ensemble_by_id(self, ensemble_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ensemble = api.get_ensemble(ensemble_id)
            if ensemble is None:
                raise werkzeug_exc.NotFound()
            resolve_ref_uri(ensemble, ensemble_id)
            return ensemble

    def realization_by_id(self, ensemble_id, realization_idx):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            realization = api.get_realization(ensemble_id, realization_idx, None)
            if realization is None:
                raise werkzeug_exc.NotFound()
            resolve_ref_uri(realization, ensemble_id)
            return realization

    def response_by_name(self, ensemble_id, response_name):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            response = api.get_response(ensemble_id, response_name, None)
            if response is None:
                raise werkzeug_exc.NotFound()
            response["alldata_ref"] = None  # value is irrelevant
            resolve_ref_uri(response, ensemble_id)
            return response

    def response_data_by_name(self, ensemble_id, response_name):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ids = api.get_response_data(ensemble_id, response_name)
            if ids is None:
                raise werkzeug_exc.NotFound()
            return self._datas(ids)

    def parameter_by_id(self, ensemble_id, parameter_def_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            parameter = api.get_parameter(ensemble_id, parameter_def_id)
            if parameter is None:
                raise werkzeug_exc.NotFound()
            parameter["alldata_ref"] = None  # value is irrelevant
            resolve_ref_uri(parameter, ensemble_id)
            return parameter

    def parameter_data_by_id(self, ensemble_id, parameter_def_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            ids = api.get_parameter_data(ensemble_id, parameter_def_id)
            if ids is None:
                raise werkzeug_exc.NotFound()
            return self._datas(ids)

    def data(self, data_id):
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            data = api.get_data(data_id)
            if data is None:
                raise werkzeug_exc.NotFound()
            if isinstance(data, list):
                return ",".join([str(x) for x in data])
            else:
                return str(data)

    def _datas(self, ids):
        def generator():
            with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
                first = True
                for data in api.get_datas(ids):
                    if first:
                        first = False
                    else:
                        yield "\n"
                    if isinstance(data, list):
                        yield ",".join([str(x) for x in data])
                    else:
                        yield str(data)

        response = Response(generator(), mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=data.csv"
        return response

    def get_observation(self, name):
        """Return an observation."""
        with StorageApi(rdb_url=self._rdb_url, blob_url=self._blob_url) as api:
            obs = api.get_observation(name)
            resolve_ref_uri(obs)
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
            if not js["attributes"]:
                raise werkzeug_exc.BadRequest()
            for k, v in js["attributes"].items():
                obs = api.set_observation_attribute(name, k, v)
                if obs is None:
                    raise werkzeug_exc.NotFound()
            return api.get_observation(name), 201

    def shutdown(self):
        request.environ.get("werkzeug.server.shutdown")()
        return "Server shutting down."


def run_server(args):
    wrapper = FlaskWrapper(
        rdb_url="sqlite:///entities.db", blob_url="sqlite:///blobs.db"
    )
    (bind_host, bind_port) = args.bind.split(":")
    from flask_cors import CORS

    cors = CORS(wrapper.app)
    wrapper.app.run(host=bind_host, port=bind_port, debug=args.debug)
