import ert_shared.storage.paths as p


def test_api(app_client):
    response = app_client.get(p.ensembles())
    ensembles = response.json()

    for ens in ensembles["ensembles"]:
        ensemble = app_client.get(p.ensemble(ens["id"])).json()

        for real in range(ensemble["realizations"]):
            realization = app_client.get(p.realization(ens["id"], real)).json()

            for response in realization["responses"]:
                assert response["data"] is not None


def test_observation(app_client):
    ens = app_client.get(p.ensemble(1)).json()
    expected = {
        "active": [True, True],
        "scale": [1, 1],
        "x_axis": [0, 3],
        "std": [1, 3],
        "values": [10.1, 10.2],
    }

    resp = app_client.get(p.response(ens["id"], ens["responses"][0]["id"])).json()
    observations = resp["observations"]

    actual = {}
    for obs in observations:
        for name, data_def in obs["data"].items():
            data = data_def["data"]
            actual[name] = data

    assert actual == expected


def test_get_single_misfits(app_client):
    ens = app_client.get(p.ensemble(1)).json()
    response = app_client.get(p.response(ens["id"], ens["responses"][0]["id"])).json()
    real_0 = response["realizations"][0]
    assert "univariate_misfits" in real_0
    assert "observation_one" in real_0["univariate_misfits"]
    misfits_0 = real_0["univariate_misfits"]["observation_one"][0]
    assert "value" in misfits_0
    assert misfits_0["obs_location"] == 0
    assert misfits_0["sign"] == True


def test_get_single_observation(app_client):
    obs = app_client.get(p.observation("observation_one")).json()

    assert obs["attributes"] == {"region": "1"}
    assert obs["name"] == "observation_one"

    x_axis = obs["x_axis"]
    assert x_axis == [0, 3]

    values = obs["values"]
    assert values == [10.1, 10.2]

    stds = obs["errors"]
    assert stds == [1, 3]


def test_get_ensemble_id_404(app_client):
    resp = app_client.get(p.observation("not_existing"))
    assert resp.status_code == 404


def test_get_single_observation_404(app_client):
    resp = app_client.get(p.observation("not_existing"))
    assert resp.status_code == 404


def test_get_observation_attributes(app_client):
    create_resp = app_client.post(
        p.observation_attributes("observation_one"),
        json={"region": "2", "depth": "9000"},
    )
    assert create_resp.status_code == 201
    obs = app_client.get(p.observation_attributes("observation_one")).json()
    expected = {"region": "2", "depth": "9000"}
    assert obs == expected


def test_parameter(app_client):
    schema = app_client.get(p.parameter(1, 3)).json()
    print(schema)
    assert schema["name"] == "key1"
    assert schema["group"] == "group"
    assert schema["prior"]["function"] == "function"


def test_get_batched_response(app_client):
    resp = _fetch_response(
        app_client, ensemble_name="ensemble_name", response_name="response_two"
    )
    data_url = p.response_data(resp["ensemble_id"], resp["id"])

    data_resp = app_client.get(data_url)

    expected = "12.1,12.2,11.1,11.2,9.9,9.3\n12.1,12.2,11.1,11.2,9.9,9.3\n"
    csv = data_resp.text
    assert expected == csv


def test_get_batched_response_missing(app_client):
    data_url = p.response_data(1, "none")
    data_resp = app_client.get(data_url)
    assert data_resp.status_code == 404


def test_get_batched_parameter(app_client):
    param = _fetch_parameter(
        app_client,
        ensemble_name="ensemble_name",
        parameter_name="A",
        parameter_group="G",
    )
    data_url = p.parameter_data(param["ensemble_id"], param["id"])

    data_resp = app_client.get(data_url)

    expected = "0,1\n1,1\n"
    csv = data_resp.text
    assert expected == csv


def test_get_batched_parameter_missing(app_client):
    resp = app_client.get(p.parameter_data(1, 42))
    assert resp.status_code == 404


def _fetch_ensemble(app_client, ensemble_name):
    return app_client.get(p.ensemble(ensemble_name)).json()


def _fetch_response(app_client, ensemble_name, response_name):
    ens = _fetch_ensemble(app_client, ensemble_name)
    resp = next(resp for resp in ens["responses"] if resp["name"] == response_name)
    response_url = p.response(ens["id"], resp["id"])
    return app_client.get(response_url).json()


def _fetch_parameter(app_client, ensemble_name, parameter_name, parameter_group):
    ens = _fetch_ensemble(app_client, ensemble_name)
    resp = next(
        param
        for param in ens["parameters"]
        if param["key"] == parameter_name and param["group"] == parameter_group
    )
    return app_client.get(p.response(ens["id"], resp["id"])).json()
