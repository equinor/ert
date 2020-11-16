import os
import flask
import yaml
import random
import string
import json
import sys
import socket
import datetime
from ert_shared.storage.storage_api import StorageApi
from pathlib import Path
from flask import Response, request, abort, jsonify
from gunicorn.app.base import BaseApplication
from ert_shared.storage import ERT_STORAGE, connection
from contextlib import contextmanager


def generate_authtoken():
    chars = string.ascii_letters + string.digits
    return "".join([random.choice(chars) for _ in range(16)])


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
                elif type_name == "alldata":
                    struct["alldata_url"] = "{}/data".format(request.url)
                else:
                    continue
                del struct[key]
            else:
                resolve_ref_uri(val, ensemble_id)


class FlaskWrapper:
    def __init__(self, secure=True, url=None):
        ERT_STORAGE.initialize(url=url)
        app = flask.Flask("ert http api")
        self.app = app

        if secure:
            self.authtoken = generate_authtoken()

            @app.before_request
            def check_auth():
                if request.authorization is None:
                    abort(401)
                un = request.authorization["username"]
                pw = request.authorization["password"]
                if un != "__token__" or pw != self.authtoken:
                    abort(401)

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
            "/schema.json",
            "schema",
            self.schema,
            methods=["GET"],
        )

        @app.route("/healthcheck")
        def healthcheck():
            return jsonify({"date": datetime.datetime.now().isoformat()})

    def schema(self):
        cur_path = Path(__file__).parent
        schema_file = cur_path / "oas.yml"
        with open(schema_file) as f:
            return yaml.safe_load(f)

    def ensembles(self):
        with self.session() as api:
            ensembles = api.get_ensembles()
            resolve_ref_uri(ensembles)
            return ensembles

    def ensemble_by_id(self, ensemble_id):
        with self.session() as api:
            ensemble = api.get_ensemble(ensemble_id)
            if ensemble is None:
                abort(404)
            resolve_ref_uri(ensemble, ensemble_id)
            return ensemble

    def realization_by_id(self, ensemble_id, realization_idx):
        with self.session() as api:
            realization = api.get_realization(ensemble_id, realization_idx, None)
            if realization is None:
                abort(404)
            resolve_ref_uri(realization, ensemble_id)
            return realization

    def response_by_name(self, ensemble_id, response_name):
        with self.session() as api:
            response = api.get_response(ensemble_id, response_name, None)
            if response is None:
                abort(404)
            response["alldata_ref"] = None  # value is irrelevant
            resolve_ref_uri(response, ensemble_id)
            return response

    def response_data_by_name(self, ensemble_id, response_name):
        with self.session() as api:
            ids = api.get_response_data(ensemble_id, response_name)
            if ids is None:
                abort(404)
            return self._datas(ids)

    def parameter_by_id(self, ensemble_id, parameter_def_id):
        with self.session() as api:
            parameter = api.get_parameter(ensemble_id, parameter_def_id)
            if parameter is None:
                abort(404)
            parameter["alldata_ref"] = None  # value is irrelevant
            resolve_ref_uri(parameter, ensemble_id)
            return parameter

    def parameter_data_by_id(self, ensemble_id, parameter_def_id):
        with self.session() as api:
            ids = api.get_parameter_data(ensemble_id, parameter_def_id)
            if ids is None:
                abort(404)
            return self._datas(ids)

    def _datas(self, datas):
        def generator():
            first = True
            for data in datas:
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
        with self.session() as api:
            obs = api.get_observation(name)
            resolve_ref_uri(obs)
            if obs is None:
                abort(404)
            return obs

    def get_observation_attributes(self, name):
        """Return attributes for an observation.

        {
            "attributes": {...}
        }
        """
        with self.session() as api:
            attrs = api.get_observation_attributes(name)
            if attrs is None:
                abort(404)
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
        with self.session() as api:
            js = request.get_json()
            if not js["attributes"]:
                abort(400)
            for k, v in js["attributes"].items():
                obs = api.set_observation_attribute(name, k, v)
                if obs is None:
                    abort(404)
            return api.get_observation(name), 201

    def shutdown(self):
        request.environ.get("werkzeug.server.shutdown")()
        return "Server shutting down."

    @contextmanager
    def session(self):
        """Provide a transactional scope around a series of operations."""
        session = ERT_STORAGE.Session()
        try:
            yield StorageApi(session)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()


class Application(BaseApplication):
    def __init__(self, wrapper, lockfile):
        self.wrapper = wrapper
        self.lockfile = lockfile
        super().__init__()

    def load_config(self):
        self.cfg.set("bind", ["0.0.0.0:0"])
        self.cfg.set("when_ready", self.when_ready)
        self.cfg.set("on_exit", self.on_exit)

    def load(self):
        return self.wrapper.app

    def when_ready(self, server):
        assert len(server.LISTENERS) == 1
        sock = server.LISTENERS[0].sock
        bind_host, bind_port = sock.getsockname()[:2]

        print(f"Started on {bind_host}:{bind_port}")
        print(f"Authtoken '{self.wrapper.authtoken}'")
        print(f"To test:")
        print(
            f"curl -u __token__:{self.wrapper.authtoken} {bind_host}:{bind_port}/ensembles"
        )

        connection_info = json.dumps(
            {
                "urls": [
                    f"http://{host}:{bind_port}"
                    for host in (
                        "127.0.0.1",
                        socket.gethostname(),
                        socket.getfqdn(),
                    )
                ],
                "authtoken": self.wrapper.authtoken,
            }
        )

        if self.lockfile:
            self.lockfile.write_text(connection_info)
            connection.set_global_info(os.getcwd())

        fd = os.environ.get("ERT_COMM_FD")
        if fd is not None:
            with os.fdopen(int(fd), "w") as fo:
                fo.write(connection_info)

    def on_exit(self, server):
        if self.lockfile:
            self.lockfile.unlink()


def parse_args():
    from argparse import ArgumentParser
    from ert_shared.storage.command import add_parser_options

    ap = ArgumentParser()
    add_parser_options(ap)
    return ap.parse_args()


def terminate_on_parent_death():
    """Quit the server when the parent does a SIGABRT or is otherwise destroyed.
    This functionality has existed on Linux for a good while, but it isn't
    exposed in the Python standard library. Use ctypes to hook into the
    functionality.
    """
    if sys.platform != "linux" or "ERT_COMM_FD" not in os.environ:
        return

    from ctypes import CDLL, c_int, c_ulong
    import signal

    lib = CDLL(None)

    # from <sys/prctl.h>
    # int prctl(int option, ...)
    prctl = lib.prctl
    prctl.restype = c_int
    prctl.argtypes = (c_int, c_ulong)

    # from <linux/prctl.h>
    PR_SET_PDEATHSIG = 1

    # connect parent death signal to our SIGTERM
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def run_server(args=None):
    if args is None:
        args = parse_args()

    wrapper = FlaskWrapper(url=args.rdb_url)

    runpath = Path(args.runpath)
    assert runpath.is_dir()

    lock = None
    if not args.disable_lockfile:
        lock = runpath / "storage_server.json"
        if lock.exists():
            raise RuntimeError("storage_server.json already exists")

    terminate_on_parent_death()
    Application(wrapper, lock).run()


if __name__ == "__main__":
    run_server()
