import gc
import io
import os
from urllib.parse import quote
from uuid import UUID

import hypothesis.strategies as st
import pandas as pd
import polars
import pytest
from hypothesis import assume, reproduce_failure
from hypothesis.stateful import rule
from starlette.testclient import TestClient

from ert.dark_storage import enkf
from ert.dark_storage.app import app
from tests.ert.unit_tests.storage.test_local_storage import StatefulStorageTest


def escape(s):
    return quote(quote(quote(s, safe="")))


class DarkStorageStateTest(StatefulStorageTest):
    def __init__(self):
        super().__init__()
        self.prev_no_token = os.environ.get("ERT_STORAGE_NO_TOKEN")
        self.prev_ens_path = os.environ.get("ERT_STORAGE_ENS_PATH")
        os.environ["ERT_STORAGE_NO_TOKEN"] = "yup"
        os.environ["ERT_STORAGE_ENS_PATH"] = str(self.storage.path)
        self.client = TestClient(app)

    @rule()
    def get_experiments_through_client(self):
        self.client.get("/updates/storage")
        response = self.client.get("/experiments")
        experiment_records = response.json()
        assert len(experiment_records) == len(list(self.storage.experiments))
        for record in experiment_records:
            storage_experiment = self.storage.get_experiment(UUID(record["id"]))
            assert {UUID(i) for i in record["ensemble_ids"]} == {
                ens.id for ens in storage_experiment.ensembles
            }

    @reproduce_failure(
        "6.122.1",
        b"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",
    )
    @rule(model_ensemble=StatefulStorageTest.ensembles)
    def get_observations_through_client(self, model_ensemble):
        model_experiment = self.storage.get_ensemble(model_ensemble.uuid).experiment
        obs_keys = {
            name: key
            for name, ds in model_experiment.observations.items()
            for key in ds["observation_key"]
        }
        for response_name in obs_keys:
            response = self.client.get(
                f"/ensembles/{model_ensemble.uuid}/records/{response_name}/observations"
            )
            assert {r["name"] for r in response.json()} == list(obs_keys.values())

    @rule(model_experiment=StatefulStorageTest.experiments)
    def get_ensembles_through_client(self, model_experiment):
        response = self.client.get(f"/experiments/{model_experiment.uuid}/ensembles")
        assert {r["id"] for r in response.json()} == {
            str(uuid) for uuid in model_experiment.ensembles
        }

    @rule(model_ensemble=StatefulStorageTest.ensembles)
    def get_responses_through_client(self, model_ensemble):
        response = self.client.get(f"/ensembles/{model_ensemble.uuid}/responses")
        response_names = {
            k
            for r in model_ensemble.response_values.values()
            for k in r["response_key"]
        }
        assert set(response.json().keys()) == response_names

    @rule(model_ensemble=StatefulStorageTest.ensembles, data=st.data())
    def get_response_csv_through_client(self, model_ensemble, data):
        assume(model_ensemble.response_values)
        response_name, response_key = data.draw(
            st.sampled_from(
                [
                    (response_name, response_key)
                    for response_name, r in model_ensemble.response_values.items()
                    for response_key in r["response_key"]
                ]
            )
        )
        df = pd.read_parquet(
            io.BytesIO(
                self.client.get(
                    f"/ensembles/{model_ensemble.uuid}/records/{escape(response_key)}",
                    headers={"accept": "application/x-parquet"},
                ).content
            )
        )
        assert {dt[:10] for dt in df.columns} == {
            str(dt)[:10]
            for dt in model_ensemble.response_values[response_name].filter(
                polars.col("response_key") == response_key
            )["time"]
        }

    def teardown(self):
        super().teardown()
        if enkf._storage is not None:
            enkf._storage.close()
        enkf._storage = None
        gc.collect()
        if self.prev_no_token is not None:
            os.environ["ERT_STORAGE_NO_TOKEN"] = self.prev_no_token
        else:
            del os.environ["ERT_STORAGE_NO_TOKEN"]
        if self.prev_ens_path is not None:
            os.environ["ERT_STORAGE_ENS_PATH"] = self.prev_ens_path
        else:
            del os.environ["ERT_STORAGE_ENS_PATH"]


TestDarkStorage = pytest.mark.integration_test(DarkStorageStateTest.TestCase)
