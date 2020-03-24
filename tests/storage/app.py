from ert_shared.storage.storage_api import StorageApi
import flask
from flask import Response 
from flask import request


def resolve_ensemble_uri(ref_pointer):
    BASE_URL = request.host_url
    return "{}/ensembles/{}".format(BASE_URL, ref_pointer)

class FlaskWrapper:
    def __init__(self, session):
        self.app = flask.Flask("Ert http api")
        self.app.add_url_rule('/ensembles', 'ensembles', self.ensembles)
        #self.app.add_url_rule('/ensembles/<ensemble_id>', 'ensemble', self.ensemble_by_id)
        self.api = StorageApi(session=session, blob_session=session)
    
    def ensembles(self):
        ensembles = self.api.ensembles()    
        for index, ensemble in enumerate(ensembles):
            uri = resolve_ensemble_uri(ensemble['ref_pointer'])
            ensembles[index]['ref_pointer'] = uri
        return Response(ensembles, "text/plain")

    def ensemble_by_id(self, ensemble_id):
        return Response(self.api.ensemble_schema(ensemble_id), "text/plain")
        

# # Flask provides a way to test your application by exposing the Werkzeug test Client
# # and handling the context locals for you.
# testing_client = flask_app.test_client()
# # Establish an application context before running the tests.
# ctx = flask_app.app_context()
# ctx.push()
# yield testing_client
# ctx.pop()