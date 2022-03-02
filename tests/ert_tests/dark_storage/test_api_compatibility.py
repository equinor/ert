from typing import List
import pandas as pd
import io

from requests import Response


def test_openapi(ert_storage_app, dark_storage_app):
    """
    Test that the openapi.json of Dark Storage is identical to ERT Storage
    """
    expect = ert_storage_app.openapi()
    actual = dark_storage_app.openapi()

    # Remove /gql endpoint, as this is a workaround to get EnkfMain injected in the request object
    del actual["paths"]["/gql"]
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


def test_graphql(env):
    from ert_storage.graphql import schema as ert_schema
    from ert_shared.dark_storage.graphql import schema as dark_schema
    from graphql import print_schema

    def _sort_schema(schema: str) -> str:
        """
        Assuming that each block is separated by an empty line, we sort the contents
        so that the order is irrelevant
        """
        sorted_blocks: List[str] = []
        for block in schema.split("\n\n"):
            lines = block.splitlines()
            if len(lines) == 1:  # likely a lone "Scalar SomeType"
                sorted_blocks.append(block)
                continue
            body = sorted(
                line for line in lines[1:-1] if "Pk:" not in line and " pk:" not in line
            )
            sorted_blocks.append("\n".join([lines[0], *body, lines[-1]]))
        return "\n\n".join(sorted_blocks)

    expect = _sort_schema(print_schema(ert_schema))
    actual = _sort_schema(print_schema(dark_schema))

    assert expect == actual


def test_response_comparison(run_poly_example_new_storage):
    new_storage_client, dark_storage_client = run_poly_example_new_storage

    # Compare ensembles
    new_storage_resp: Response = new_storage_client.post(
        "/gql",
        json={"query": "{experiments{ensembles{size, userdata, activeRealizations}}}"},
    )
    dark_storage_resp: Response = dark_storage_client.post(
        "/gql",
        json={"query": "{experiments{ensembles{size, userdata, activeRealizations}}}"},
    )
    assert new_storage_resp.json() == dark_storage_resp.json()

    # Compare responses
    new_storage_responses: Response = new_storage_client.post(
        "/gql",
        json={
            "query": "{experiments{ensembles{responses{name, realizationIndex, userdata}}}}"
        },
    )

    dark_storage_responses: Response = dark_storage_client.post(
        "/gql",
        json={
            "query": "{experiments{ensembles{responses{name, realizationIndex, userdata}}}}"
        },
    )

    # Drop @ form response name if it is there
    # Dark storage gen data response names contain @0
    ds_response_json = dark_storage_responses.json()
    for ens in ds_response_json["data"]["experiments"][0]["ensembles"]:
        for resp in ens["responses"]:
            resp["name"] = resp["name"].split("@")[0]

    assert new_storage_responses.json() == ds_response_json

    # Compare responses data
    def get_resp_dataframe(resp):
        stream = io.BytesIO(resp.content)
        return pd.read_csv(stream, index_col=0, float_precision="round_trip")

    ds_dfs = []
    ns_dfs = []

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
        ns_resp: Response = new_storage_client.post(
            "/gql",
            json={
                "query": f'{{ensemble(id: "{ens_id_ns}") {{parameters {{name}} }} }}'
            },
        )
        ds_resp: Response = dark_storage_client.post(
            "/gql",
            json={
                "query": f'{{ensemble(id: "{ens_id_ds}") {{parameters {{name}} }} }}'
            },
        )

        assert ns_resp.json() == ds_resp.json()

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
