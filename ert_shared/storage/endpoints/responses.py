from fastapi import APIRouter
from fastapi.responses import Response
from typing import List
from datetime import datetime
from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds

import pandas as pd


router = APIRouter()


def _calculate_misfit(obs_value, response_value, obs_std, x_value):
    difference = response_value - obs_value
    misfit = (difference / obs_std) ** 2
    sign = difference > 0

    return {"value": misfit, "sign": sign, "obs_location": x_value}


def _obs_to_json(obs, transformation):
    data = {
        "name": obs.name,
        "data": {
            "values": {"data": obs.values},
            "std": {"data": obs.errors},
            "x_axis": {"data": obs.x_axis},
            # Defaults to all active and scale of 1 if no transformation
            "active": {"data": [True for _ in obs.x_axis]},
            "scale": {"data": [1 for _ in obs.x_axis]},
        },
    }
    if transformation is not None:
        data["data"]["active"] = ({"data": transformation.active_list},)
        data["data"]["scale"] = ({"data": transformation.scale_list},)

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
    exists = (
        db.query(ds.ResponseDefinition.id)
        .filter_by(ensemble_id=ensemble_id, name=resp.name)
        .count()
        > 0
    )
    if not exists:
        obj = ds.ResponseDefinition(
            name=resp.name, indices=resp.indices, ensemble_id=ensemble_id
        )
        db.add(obj)
    else:
        obj = (
            db.query(ds.ResponseDefinition)
            .filter_by(ensemble_id=ensemble_id, name=resp.name)
            .one()
        )
        obj.indices = resp.indices

    db.add_all(
        ds.Response(values=values, index=index, response_definition=obj)
        for index, values in resp.realizations.items()
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
        univariate_misfits[resp.index] = {}
        response_xaxis = list(bundle.indices)

        for link in observation_links:
            observation = link.observation
            obs_values = list(observation.values)
            obs_stds = list(observation.errors)
            obs_xaxis = list(observation.x_axis)

            misfits = []
            for obs_index, obs_value in enumerate(obs_values):
                obs_std = obs_stds[obs_index]
                obs_x = obs_xaxis[obs_index]
                if type(obs_x) == datetime:
                    obs_x = obs_x.isoformat()
                resp_index = response_xaxis.index(obs_x)
                misfits.append(
                    _calculate_misfit(
                        obs_value=obs_value,
                        obs_std=obs_std,
                        response_value=resp_values[resp_index],
                        x_value=obs_x,
                    )
                )
            univariate_misfits[resp.index][observation.name] = misfits

    return_schema = {
        "id": id,
        "name": bundle.name,
        "ensemble_id": ensemble_id,
        "realizations": [
            {
                "name": resp.index,
                "data": resp.values,
                "summarized_misfits": {
                    misfit.observation_response_definition_link.observation.name: misfit.value
                    for misfit in resp.misfits
                },
                "univariate_misfits": {
                    obs_name: misfits
                    for obs_name, misfits in univariate_misfits[resp.index].items()
                },
            }
            for resp in responses
        ],
        "axis": {"data": bundle.indices},
    }

    if len(observation_links) > 0:
        transformations = (
            db.query(ds.ObservationTransformation)
            .filter(
                ds.ObservationTransformation.observation_id.in_(
                    link.observation.id for link in observation_links
                )
            )
            .join(ds.ObservationTransformation.update)
            .filter_by(ensemble_result_id=ensemble_id)
            .all()
        )

        # All connected transformations should in the future be compressed to one
        # scale vector and one active vector when serving the data back to the user
        transformations = {
            transformation.observation_id: transformation
            for transformation in transformations
        }

        return_schema["observations"] = [
            _obs_to_json(link.observation, transformations.get(link.observation_id))
            for link in observation_links
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
