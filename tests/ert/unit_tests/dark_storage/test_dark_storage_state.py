import io
import os
from urllib.parse import quote
from uuid import UUID

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import assume
from hypothesis.stateful import rule
from starlette.testclient import TestClient

from ert.dark_storage.app import app
from ert.dark_storage.enkf import update_storage
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
        update_storage()
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

    @rule(model_experiment=StatefulStorageTest.experiments)
    def get_observations_through_client(self, model_experiment):
        response = self.client.get(f"/experiments/{model_experiment.uuid}/observations")
        assert {r["name"] for r in response.json()} == {
            key
            for _, ds in model_experiment.observations.items()
            for key in ds["observation_key"]
        }

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
        response_key, response_name = data.draw(
            st.sampled_from(
                [
                    (key, n)
                    for key, r in model_ensemble.response_values.items()
                    for n in r["name"].values
                ]
            )
        )
        df = pd.read_parquet(
            io.BytesIO(
                self.client.get(
                    f"/ensembles/{model_ensemble.uuid}/records/{escape(response_name)}",
                    headers={"accept": "application/x-parquet"},
                ).content
            )
        )
        assert {dt[:10] for dt in df.columns} == {
            str(dt)[:10]
            for dt in model_ensemble.response_values[response_key]
            .sel(name=response_name)["time"]
            .values
        }

    def teardown(self):
        super().teardown()
        if self.prev_no_token is not None:
            os.environ["ERT_STORAGE_NO_TOKEN"] = self.prev_no_token
        else:
            del os.environ["ERT_STORAGE_NO_TOKEN"]
        if self.prev_ens_path is not None:
            os.environ["ERT_STORAGE_ENS_PATH"] = self.prev_ens_path
        else:
            del os.environ["ERT_STORAGE_ENS_PATH"]


TestDarkStorage = pytest.mark.integration_test(DarkStorageStateTest.TestCase)
