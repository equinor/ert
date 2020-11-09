from fastapi import APIRouter
from fastapi.responses import Response
from typing import List

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds

import pandas as pd


router = APIRouter()


def _calculate_misfit(
    obs_value, response_values, obs_stds, obs_data_indexes, obs_index
):
    observation_std = obs_stds[obs_index]
    response_index = obs_data_indexes[obs_index]
    response_value = response_values[response_index]
    difference = response_value - obs_value
    misfit = (difference / observation_std) ** 2
    sign = difference > 0

    return {"value": misfit, "sign": sign, "obs_index": obs_index}


def _obs_to_json(obs, active=None):
    data = {
        "name": obs.name,
        "data": {
            "values": {"data": obs.values},
            "std": {"data": obs.errors},
            "data_indexes": {"data": obs.data_indices},
            "key_indexes": {"data": obs.key_indices},
        },
    }
    if active is not None:
        data["data"]["active_mask"] = {"data": active}

    attrs = obs.get_attributes()
    if len(attrs) > 0:
        data["attributes"] = attrs

    return data


@router.get("/ensembles/{ensemble_id}/responses", response_model=List[js.Response])
async def read_responses(*, db: Session = Db(), ensemble_id: int):
    pass


@router.post("/ensembles/{ensemble_id}/responses")
async def create_responses(
    *, db: Session = Db(), ensemble_id: int, resp: js.ResponseCreate
):
    obj = ds.ResponseDefinition(
        name=resp.name, indices=resp.indices, ensemble_id=ensemble_id
    )
    db.add(obj)
    db.add_all(
        ds.Response(values=values, realization_id=real.id, response_definition=obj)
        for real, values in (
            (
                (
                    db.query(ds.Realization)
                    .filter_by(ensemble_id=ensemble_id, index=index)
                    .one()
                ),
                values,
            )
            for index, values in resp.realizations.items()
        )
    )
    db.commit()


@router.get("/ensembles/{ensemble_id}/responses/{id}/data")
async def read_responses_as_csv(*, db: Session = Db(), ensemble_id: int, id: int):
    df = pd.DataFrame(
        [
            ref.values
            for ref in (
                db.query(ds.Response.values)
                .filter(ds.Response.response_definition_id == id)
                .all()
            )
        ]
    )
    return Response(content=df.to_csv(index=False, header=False), media_type="text/csv")


@router.get("/ensembles/{ensemble_id}/responses/{id}")
async def read_response_by_id(*, db: Session = Db(), ensemble_id: int, id: int):
    bundle = (
        db.query(ds.ResponseDefinition)
        .filter(
            ds.ResponseDefinition.id == id,
            ds.ResponseDefinition.ensemble_id == ensemble_id,
        )
        .one()
    )

    observation_links = bundle.observation_links
    responses = bundle.responses
    univariate_misfits = {}
    for resp in responses:
        resp_values = list(resp.values)
        univariate_misfits[resp.realization.index] = {}
        for link in observation_links:
            observation = link.observation
            obs_values = list(observation.values)
            obs_stds = list(observation.errors)
            obs_data_indexes = list(observation.data_indices)
            misfits = []
            for obs_index, obs_value in enumerate(obs_values):
                misfits.append(
                    _calculate_misfit(
                        obs_value,
                        resp_values,
                        obs_stds,
                        obs_data_indexes,
                        obs_index,
                    )
                )
            univariate_misfits[resp.realization.index][observation.name] = misfits

    return_schema = {
        "id": id,
        "name": bundle.name,
        "ensemble_id": ensemble_id,
        "realizations": [
            {
                "name": resp.realization.index,
                "realization_ref": resp.realization.index,
                "data": resp.values,
                "summarized_misfits": {
                    misfit.observation_response_definition_link.observation.name: misfit.value
                    for misfit in resp.misfits
                },
                "univariate_misfits": {
                    obs_name: misfits
                    for obs_name, misfits in univariate_misfits[
                        resp.realization.index
                    ].items()
                },
            }
            for resp in responses
        ],
        "axis": {"data": bundle.indices},
    }

    if len(observation_links) > 0:
        return_schema["observations"] = [
            _obs_to_json(link.observation, link.active) for link in observation_links
        ]

    return return_schema


@router.get("/ensembles/{ensemble_id}/responses/{id}/data")
async def read_response_data(
    *,
    db: Session = Db(),
    ensemble_id: int,
    id: int,
):
    pass
