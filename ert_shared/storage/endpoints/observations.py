from typing import List, Mapping, Optional
from fastapi import APIRouter
from fastapi.responses import Response
import pandas as pd

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds


router = APIRouter()


@router.get("/observations", response_model=List[js.Observation])
async def read_observations(*, db: Session = Db()):
    return db.query(ds.Observation).all()


@router.post("/observations")
async def create_observation(
    *,
    db: Session = Db(),
    obs: js.ObservationCreate,
):
    if db.query(ds.Observation).filter_by(name=obs.name).count() > 0:
        return

    obj = ds.Observation(
        name=obs.name,
        x_axis=obs.x_axis,
        values=obs.values,
        errors=obs.errors,
    )
    db.add(obj)

    return obj


@router.get("/observations/{id}", response_model=js.Observation)
async def read_observation_by_id(
    *,
    db: Session = Db(),
    id: int,
):
    return db.query(ds.Observation).filter_by(id=id).one()


@router.get("/observations/name/{name}", response_model=js.Observation)
async def read_observation_by_name(
    *,
    db: Session = Db(),
    name: str,
):
    return (
        db.query(ds.Observation)
        .filter_by(name=name)
        .order_by(ds.Observation.id.desc())
        .one()
    )


@router.get("/observations/{id}/data")
async def read_observation_data(
    *,
    db: Session = Db(),
    id: int,
):
    data = db.query(ds.Observation).filter_by(id=id).one()

    df = pd.DataFrame(
        data={
            "values": data.values,
            "errors": data.errors,
        },
        index=data.key_indices,
    )

    return Response(content=df.to_csv(), media_type="text/csv")


@router.get("/observations/{id}/attributes", response_model=Mapping[str, str])
async def read_observation_attributes(
    *,
    db: Session = Db(),
    id: int,
):
    return db.query(ds.ObservationAttribute).filter_by(observation_id=id).all()


@router.post("/observations/{id}/attributes")
async def create_observation_attribute(
    *,
    db: Session = Db(),
    id: int,
    attrs: Mapping[str, str],
):
    pass


@router.get("/observations/name/{name}/attributes", response_model=Mapping[str, str])
async def read_observation_attributes_by_name(
    *,
    db: Session = Db(),
    name: str,
):
    obs = db.query(ds.Observation).filter_by(name=name).one()
    return obs.get_attributes()


@router.post("/observations/name/{name}/attributes", status_code=201)
async def create_observation_attributes_by_name(
    *,
    db: Session = Db(),
    name: str,
    attrs: Mapping[str, str],
):
    obs = db.query(ds.Observation).filter_by(name=name).one()
    for key, val in attrs.items():
        obs.add_attribute(key, val)
    db.commit()
