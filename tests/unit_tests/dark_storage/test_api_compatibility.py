import io

import pandas as pd
from requests import Response


def test_openapi(ert_storage_app, dark_storage_app):
    """
    Test that the openapi.json of Dark Storage is identical to ERT Storage
    """
    expect = ert_storage_app.openapi()
    actual = dark_storage_app.openapi()

    # Remove textual data (descriptions and such) from ERT Storage's API.
    def _remove_text(data):
        if isinstance(data, dict):
            return {
                key: _remove_text(val)
                for key, val in data.items()
                if key not in ("description", "examples")
            }
        return data

    assert _remove_text(expect) == _remove_text(actual)


def test_response_comparison(run_poly_example_new_storage):
    new_storage_client, dark_storage_client = run_poly_example_new_storage

    # Compare priors
    new_storage_experiments: Response = new_storage_client.get("/experiments")
    dark_storage_experiments: Response = dark_storage_client.get("/experiments")
    new_storage_experiments_json = new_storage_experiments.json()
    dark_storage_experiments_json = dark_storage_experiments.json()
    assert len(new_storage_experiments_json) == len(dark_storage_experiments_json)
    assert (
        new_storage_experiments_json[0]["priors"]
        == dark_storage_experiments_json[0]["priors"]
    )

    # Compare responses data
    def get_resp_dataframe(resp):
        stream = io.BytesIO(resp.content)
        return pd.read_csv(stream, index_col=0, float_precision="round_trip")

    resp: Response = dark_storage_client.get("/experiments")
    experiment_json_ds = resp.json()

    resp: Response = new_storage_client.get("/experiments")
    experiment_json_ns = resp.json()

    # Compare resposnes  data
    for ens_id_ds, ens_id_ns in zip(
        experiment_json_ds[0]["ensemble_ids"], experiment_json_ns[0]["ensemble_ids"]
    ):
        resp: Response = dark_storage_client.get(f"/ensembles/{ens_id_ds}")
        ds_ensemble_json = resp.json()
        response_name = ds_ensemble_json["response_names"][0]
        resp: Response = dark_storage_client.get(
            f"/ensembles/{ens_id_ds}/responses/{response_name}/data"
        )
        ds_df = get_resp_dataframe(resp)

        resp: Response = new_storage_client.get(f"/ensembles/{ens_id_ns}")
        ns_ensemble_json = resp.json()
        response_name = ns_ensemble_json["response_names"][0]
        resp: Response = new_storage_client.get(
            f"/ensembles/{ens_id_ns}/responses/{response_name}/data"
        )
        ns_df = get_resp_dataframe(resp)
        df_diff = pd.concat([ds_df, ns_df]).drop_duplicates(keep=False)
        assert df_diff.empty

        ds_resp_names = [
            name.split("@")[0] for name in ds_ensemble_json["response_names"]
        ]
        assert (
            ds_ensemble_json["parameter_names"] == ns_ensemble_json["parameter_names"]
        )
        assert ds_resp_names == ns_ensemble_json["response_names"]
        assert ds_ensemble_json["userdata"] == ns_ensemble_json["userdata"]
        assert ds_ensemble_json["size"] == ns_ensemble_json["size"]

        # Compare paramete names
        ns_resp: Response = new_storage_client.get(f"/ensembles/{ens_id_ns}")
        ds_resp: Response = dark_storage_client.get(f"/ensembles/{ens_id_ds}")

        assert ns_resp.json()["parameter_names"] == ds_resp.json()["parameter_names"]

        # Compare missfits parameters
        ns_resp: Response = new_storage_client.get(
            f"/compute/misfits?ensemble_id={ens_id_ns}&response_name=POLY_RES"
        )

        ds_resp: Response = dark_storage_client.get(
            f"/compute/misfits?ensemble_id={ens_id_ds}&response_name=POLY_RES@0"
        )
        ns_df = get_resp_dataframe(ns_resp)
        ds_df = get_resp_dataframe(ds_resp)
        # Compare misfit dataframes
        df_diff = pd.concat([ds_df, ns_df]).drop_duplicates(keep=False)
        assert df_diff.empty

    # Compare observation
    experiment_id = experiment_json_ds[0]["id"]
    resp: Response = dark_storage_client.get(
        f"/experiments/{experiment_id}/observations"
    )
    ds_obs = resp.json()

    experiment_id = experiment_json_ns[0]["id"]
    resp: Response = new_storage_client.get(
        f"/experiments/{experiment_id}/observations"
    )
    ns_obs = resp.json()
    assert ns_obs[0]["name"] == ds_obs[0]["name"]
    assert ns_obs[0]["errors"] == ds_obs[0]["errors"]
    assert ns_obs[0]["values"] == ds_obs[0]["values"]
    assert ns_obs[0]["x_axis"] == ds_obs[0]["x_axis"]
