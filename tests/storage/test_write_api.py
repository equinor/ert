import pytest
import pandas as pd
from typing import List
from ert_shared.storage import extraction, database_schema as ds, json_schema as js


observation_data = {
    ("POLY_OBS", 0, 10): {"OBS": 2.0, "STD": 0.1},
    ("POLY_OBS", 2, 12): {"OBS": 7.1, "STD": 1.1},
    ("POLY_OBS", 4, 14): {"OBS": 21.1, "STD": 4.1},
    ("POLY_OBS", 6, 16): {"OBS": 31.8, "STD": 9.1},
    ("POLY_OBS", 8, 18): {"OBS": 53.2, "STD": 16.1},
    ("TEST_OBS", 3, 3): {"OBS": 6, "STD": 0.1},
    ("TEST_OBS", 6, 6): {"OBS": 12, "STD": 0.2},
    ("TEST_OBS", 9, 9): {"OBS": 18, "STD": 0.3},
}

coeff_a = pd.DataFrame.from_dict(
    {
        "COEFFS:COEFF_A": {
            0: 0.7684484807065148,
            1: 0.031542101926117616,
            2: 0.9116906743615176,
            3: 0.6985513230581486,
            4: 0.5949261230249001,
        },
    },
)

parameters = {"COEFFS:COEFF_A": coeff_a}

poly_res = pd.DataFrame.from_dict(
    {
        0: {
            0: 2.5995,
            1: 5.203511,
            2: 9.496884000000001,
            3: 15.479619,
            4: 23.151716,
            5: 32.513175000000004,
            6: 43.563995999999996,
            7: 56.304179,
            8: 70.73372400000001,
            9: 86.852631,
        },
        1: {
            0: 4.97204,
            1: 6.23818,
            2: 8.18051,
            3: 10.79903,
            4: 14.09374,
            5: 18.06464,
            6: 22.71173,
            7: 28.035009999999996,
            8: 34.03448,
            9: 40.71014,
        },
        2: {
            0: 0.660302,
            1: 1.906593,
            2: 4.597682,
            3: 8.733569000000001,
            4: 14.314254,
            5: 21.339737000000003,
            6: 29.810018000000003,
            7: 39.725097000000005,
            8: 51.084974,
            9: 63.889649000000006,
        },
        3: {
            0: 4.99478,
            1: 5.566743000000001,
            2: 6.4375800000000005,
            3: 7.607291,
            4: 9.075876000000001,
            5: 10.843335,
            6: 12.909668,
            7: 15.274875,
            8: 17.938955999999997,
            9: 20.901911,
        },
        4: {
            0: 1.96728,
            1: 2.2754027,
            2: 3.0749214,
            3: 4.3658361,
            4: 6.1481468,
            5: 8.421853500000001,
            6: 11.186956200000001,
            7: 14.4434549,
            8: 18.1913496,
            9: 22.430640299999997,
        },
    }
)

responses = {"POLY_RES": poly_res}


class MockObsBlock:
    def __init__(self, name):
        self._name = name

    def get_obs_key(self):
        return self._name

    def is_active(self, index):
        assert index in range(5)
        return True

    def __len__(self):
        return 5


class MockObsData:
    def get_num_blocks(self):
        return 2

    def get_block(self, index):
        return MockObsBlock(["POLY_OBS", "TEST_OBS"][index])


class MockUpdateStep:
    def get_obs_data(self):
        return MockObsData()


class MockObservation:
    def __init__(self, name, response_name):
        self._name = name
        self._response_name = response_name

    def getObservationKey(self):
        return self._name

    def getDataKey(self):
        return self._response_name

    def getTotalChi2(self, _fs, index):
        return float(index)


class MockEnkfObs:
    def __init__(self, obs_dict):
        self.obs_dict = obs_dict

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.obs_dict[key]
        else:
            return list(self.obs_dict.values())[key]

    def getTypedKeylist(self, type):
        return []

    def keys(self):
        return self.obs_dict.keys()


class MockFacade:
    ENSEMBLE_NAME = "default"

    def __init__(self):
        self.enkf_main = None
        self.observations = MockEnkfObs(
            {
                "POLY_OBS": MockObservation("POLY_OBS", "POLY_RES"),
                "TEST_OBS": MockObservation("TEST_OBS", "TEST_RES"),
            }
        )

    def get_observations(self):
        return self.observations

    def get_ensemble_size(self):
        return 5

    def get_observation_key(self, i):
        return list(self.observations.keys())[i]

    def get_current_case_name(self):
        return self.ENSEMBLE_NAME

    def get_current_fs(self):
        return

    def get_update_step(self):
        return [MockUpdateStep()]

    def gen_kw_priors(self):
        return {
            "COEFFS": [
                {
                    "key": "COEFF_A",
                    "function": "UNIFORM",
                    "parameters": {"MIN": 0.0, "MAX": 1.0},
                },
                {
                    "key": "COEFF_B",
                    "function": "UNIFORM",
                    "parameters": {"MIN": 0.0, "MAX": 2.0},
                },
                {
                    "key": "COEFF_C",
                    "function": "UNIFORM",
                    "parameters": {"MIN": 0.0, "MAX": 5.0},
                },
            ]
        }

    def all_data_type_keys(self):
        return set(parameters.keys()) | set(responses.keys())

    def is_gen_kw_key(self, key):
        return key in set(parameters.keys())

    def gather_gen_kw_data(self, ensemble_name, key):
        assert ensemble_name == self.ENSEMBLE_NAME
        return parameters[key]

    def is_summary_key(self, key):
        """Arbitary decision to make all responses GEN_DATAs."""
        return False

    def is_gen_data_key(self, key):
        """Arbitary decision to make all responses GEN_DATAs"""
        return key in set(responses.keys())

    def gather_gen_data_data(self, case, key):
        assert case == self.ENSEMBLE_NAME
        return responses[key]

    def gather_summary_data(self, case, key):
        raise NotImplementedError("Mock data contains no summary data")


@pytest.fixture
def mock_ert(monkeypatch):
    class MockMeasuredData:
        def __init__(self, _facade, _keys, load_data):
            assert _keys == ["POLY_OBS", "TEST_OBS"]

        def remove_inactive_observations(self):
            pass

        @property
        def data(self):
            return pd.DataFrame.from_dict(observation_data)

    def create_active_list(_enkf_main, _fs):
        return range(5)

    monkeypatch.setattr(extraction, "MeasuredData", MockMeasuredData)
    monkeypatch.setattr(
        extraction.MisfitCollector, "createActiveList", create_active_list
    )
    yield MockFacade()


def test_create_observations(app_client, mock_ert):
    observations, _ = extraction.create_observations(mock_ert)
    for obs in observations:
        app_client.post("/observations", data=obs.json())

    poly_obs = app_client.db.query(ds.Observation).filter_by(name="POLY_OBS").one()
    assert poly_obs.id is not None
    assert poly_obs.x_axis == [0, 2, 4, 6, 8]
    assert poly_obs.values == [2.0, 7.1, 21.1, 31.8, 53.2]
    assert poly_obs.errors == [0.1, 1.1, 4.1, 9.1, 16.1]

    test_obs = app_client.db.query(ds.Observation).filter_by(name="TEST_OBS").one()
    assert test_obs.id is not None
    assert test_obs.x_axis == [3, 6, 9]
    assert test_obs.values == [6, 12, 18]
    assert test_obs.errors == [0.1, 0.2, 0.3]


def test_ensemble_return(app_client, mock_ert):
    """POSTing an ensemble that is successfully created should respond with a valid
    Ensemble object
    """
    ens_in = extraction.create_ensemble(mock_ert, update_id=None)
    resp = app_client.post("/ensembles", data=ens_in.json())
    ens_out = js.Ensemble.parse_obj(resp.json())

    assert ens_out.id is not None
    assert ens_in.name == ens_out.name


def test_parameters(app_client, mock_ert):
    """Create an ensemble and test that the parameters were created correctly"""
    ensemble = extraction.create_ensemble(mock_ert, None)  # No reference
    resp = app_client.post("/ensembles", data=ensemble.json()).json()

    def get_parameter_values(group: str, name: str) -> List[float]:
        return (
            app_client.db.query(ds.Parameter.values)
            .filter_by(group=group, name=name, ensemble_id=resp["id"])
            .one()
            .values
        )

    assert get_parameter_values("COEFFS", "COEFF_A") == [
        0.7684484807065148,
        0.031542101926117616,
        0.9116906743615176,
        0.6985513230581486,
        0.5949261230249001,
    ]


def test_priors(app_client, mock_ert):
    ensemble = extraction.create_ensemble(mock_ert, None)  # No reference
    ens_resp = app_client.post("/ensembles", data=ensemble.json()).json()

    def get_prior(group, key):
        return (
            app_client.db.query(ds.ParameterPrior)
            .filter_by(group=group, key=key)
            .join(ds.ParameterPrior.ensemble)
            .filter(ds.Ensemble.id == ens_resp["id"])
            .one()
        )

    prior_1 = get_prior("COEFFS", "COEFF_A")
    assert prior_1.function == "UNIFORM"

    prior_2 = get_prior("COEFFS", "COEFF_B")
    assert prior_1.function == "UNIFORM"

    prior_3 = get_prior("COEFFS", "COEFF_C")
    assert prior_1.function == "UNIFORM"


def test_responses(app_client, mock_ert):
    ensemble = extraction.create_ensemble(mock_ert, update_id=None)
    ens_resp = app_client.post("/ensembles", data=ensemble.json()).json()

    observations, _ = extraction.create_observations(mock_ert)
    for obs in observations:
        app_client.post("/observations", data=obs.json())

    responses = extraction.create_responses(mock_ert, "default")
    for r in responses:
        app_client.post(f"/ensembles/{ens_resp['id']}/responses", data=r.json())

    response_0 = (
        app_client.db.query(ds.Response.values)
        .filter_by(index=0)
        .join(ds.Response.response_definition)
        .filter_by(ensemble_id=ens_resp["id"], name="POLY_RES")
        .one()
    )
    response_values = response_0.values
    assert response_values == [
        2.5995,
        5.203511,
        9.496884000000001,
        15.479619,
        23.151716,
        32.513175000000004,
        43.563995999999996,
        56.304179,
        70.73372400000001,
        86.852631,
    ]


def _create_dummy_update_id(algorithm, ensemble_id, app_client):
    update_resp = app_client.post(
        f"/ensembles/{ensemble_id}/updates",
        data=js.UpdateCreate(
            ensemble_reference_id=ensemble_id,
            algorithm=algorithm,
            observation_transformations=[],
        ).json(),
    )
    return js.Update.parse_obj(update_resp.json()).id


def test_ensemble_parent_child_link(app_client, mock_ert):
    ensemble_0 = extraction.create_ensemble(mock_ert, update_id=None)
    ens_resp_0 = app_client.post("/ensembles", data=ensemble_0.json())
    ens = js.Ensemble.parse_obj(ens_resp_0.json())
    assert ens_resp_0.status_code == 200

    update_ens1_id = _create_dummy_update_id("bogosort", ens.id, app_client)

    mock_ert.ENSEMBLE_NAME = "default_1"
    ensemble_1 = extraction.create_ensemble(mock_ert, update_id=update_ens1_id)
    ens_resp_1 = app_client.post("/ensembles", data=ensemble_1.json())
    assert ens_resp_1.status_code == 200

    update_ens2_id = _create_dummy_update_id("dijkstra", ens.id, app_client)
    mock_ert.ENSEMBLE_NAME = "default_2"
    ensemble_2 = extraction.create_ensemble(mock_ert, update_id=update_ens2_id)
    ens_resp_2 = app_client.post("/ensembles", data=ensemble_2.json())
    assert ens_resp_2.status_code == 200

    def get_ensemble(name):
        return (
            app_client.db.query(ds.Ensemble)
            .filter_by(name=name)
            .order_by(ds.Ensemble.id.desc())
            .first()
        )

    ens_0 = get_ensemble("default")
    ens_1 = get_ensemble("default_1")
    ens_2 = get_ensemble("default_2")

    assert ens_0.parent is None
    assert len(ens_0.children) == 2
    assert ens_0.children[0].ensemble_result == ens_1
    assert ens_0.children[1].ensemble_result == ens_2

    assert ens_1.parent.algorithm == "bogosort"
    assert ens_1.parent.ensemble_reference == ens_0
    assert ens_1.children == []

    assert ens_2.parent.algorithm == "dijkstra"
    assert ens_2.parent.ensemble_reference == ens_0
    assert ens_2.children == []


def test_update(app_client, mock_ert):
    # Setup
    ensemble = extraction.create_ensemble(mock_ert, update_id=None)
    ens_resp = app_client.post("/ensembles", data=ensemble.json()).json()

    observations, _ = extraction.create_observations(mock_ert)
    for obs in observations:
        resp = app_client.post("/observations", data=obs.json())

    responses = extraction.create_responses(mock_ert, "default")
    for r in responses:
        resp = app_client.post(f"/ensembles/{ens_resp['id']}/responses", data=r.json())
