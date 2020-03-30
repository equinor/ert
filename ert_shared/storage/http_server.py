from flask import Flask
import json
from ert_shared.storage.storage_api import StorageApi

app = Flask(__name__)
storage_api = StorageApi()


@app.route('/ensembles')
def ensembles():
    return json.dumps(storage_api.ensembles())

@app.route('/ensembles/<int:ensemble_id>/realization/<int:realization_idx>')
def realization(ensemble_id, realization_idx):
    return json.dumps(storage_api.realization(ensemble_id=ensemble_id, realization_idx=realization_idx, filter=None))

@app.route('/data/<int:reference>')
def data(reference):
    return json.dumps(storage_api.data(id=reference).data)

def run_server(path=None): 
    app.run()