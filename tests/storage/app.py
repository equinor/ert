from ert_shared.storage.storage_api import StorageApi
import flask
from flask import Response
from flask import request


def resolve_ensemble_uri(ref_pointer):
    BASE_URL = request.host_url
    return "{}ensembles/{}".format(BASE_URL, ref_pointer)

def resolve_data_uri(struct):
    BASE_URL = request.host_url

    if isinstance(struct, list):
        for item in struct:
            resolve_data_uri(item)
    elif isinstance(struct, dict):
        for key, val in struct.items():
            if key == "data_refs":
                for name in val:
                    id = val[name]
                    url = "{}data/{}".format(BASE_URL, id)
                    val[name] = url
            else:
                resolve_data_uri(val)


class FlaskWrapper:
    def __init__(self, session):
        self.app = flask.Flask("Ert http api")
        self.app.add_url_rule('/ensembles', 'ensembles', self.ensembles)
        self.app.add_url_rule('/ensembles/<ensemble_id>', 'ensemble', self.ensemble_by_id)
        self.app.add_url_rule('/ensembles/<ensemble_id>/realizations', 'realizations', self.realizations)
        self.app.add_url_rule('/ensembles/<ensemble_id>/realizations/<realization_id>', 'realizations', self.realization_by_id)
        self.app.add_url_rule('/data/<int:data_id>', 'data', self.data)
        self.api = StorageApi(session=session, blob_session=session)
    
    def ensembles(self):
        ensembles = self.api.ensembles()

        for index, ensemble in enumerate(ensembles):
            uri = resolve_ensemble_uri(ensemble['ref_pointer'])
            ensembles[index]['ref_pointer'] = uri
        return {"ensembles": ensembles}

    def ensemble_by_id(self, ensemble_id):
        ensemble = self.api.ensemble_schema(ensemble_id)
        resolve_data_uri(ensemble)
        return ensemble

    def realizations(self, ensemble_id):
        pass

    def realizations_by_id(self, ensemble_id, realization_id):
        pass

    def data(self, data_id):
        data = self.api.data(data_id).data
        if isinstance(data, list):
            return ",".join([str(x) for x in data])
        else:
            return str(data)


# # Flask provides a way to test your application by exposing the Werkzeug test Client
# # and handling the context locals for you.
# testing_client = flask_app.test_client()
# # Establish an application context before running the tests.
# ctx = flask_app.app_context()
# ctx.push()
# yield testing_client
# ctx.pop()