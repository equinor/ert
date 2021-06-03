from uuid import UUID

from fastapi import APIRouter, Depends, Body
from typing import List, Any, Mapping, Optional
from sqlalchemy.orm.attributes import flag_modified
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js


router = APIRouter(tags=["ensemble"])


@router.post(
    "/experiments/{experiment_id}/observations", response_model=js.ObservationOut
)
def post_observation(
    *, db: Session = Depends(get_db), obs_in: js.ObservationIn, experiment_id: UUID
) -> js.ObservationOut:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    records = (
        [db.query(ds.Record).filter_by(id=rec_id).one() for rec_id in obs_in.records]
        if obs_in.records is not None
        else []
    )
    obs = ds.Observation(
        name=obs_in.name,
        x_axis=obs_in.x_axis,
        errors=obs_in.errors,
        values=obs_in.values,
        experiment=experiment,
        records=records,
    )

    db.add(obs)
    db.commit()

    return _observation_from_db(obs)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=List[js.ObservationOut]
)
def get_observations(
    *, db: Session = Depends(get_db), experiment_id: UUID
) -> List[js.ObservationOut]:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    return [_observation_from_db(obs) for obs in experiment.observations]


@router.get(
    "/ensembles/{ensemble_id}/observations", response_model=List[js.ObservationOut]
)
def get_observations_with_transformation(
    *, db: Session = Depends(get_db), ensemble_id: UUID
) -> List[js.ObservationOut]:
    ens = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    experiment = ens.experiment
    update = ens.parent
    transformations = (
        {trans.observation.name: trans for trans in update.observation_transformations}
        if update is not None
        else {}
    )

    return [
        _observation_from_db(obs, transformations) for obs in experiment.observations
    ]


@router.put("/observations/{obs_id}/userdata")
async def replace_observation_userdata(
    *,
    db: Session = Depends(get_db),
    obs_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Assign new userdata json
    """
    obs = db.query(ds.Observation).filter_by(id=obs_id).one()
    obs.userdata = body
    db.commit()


@router.patch("/observations/{obs_id}/userdata")
async def patch_observation_userdata(
    *,
    db: Session = Depends(get_db),
    obs_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Update userdata json
    """
    obs = db.query(ds.Observation).filter_by(id=obs_id).one()
    obs.userdata.update(body)
    flag_modified(obs, "userdata")
    db.commit()


@router.get("/observations/{obs_id}/userdata", response_model=Mapping[str, Any])
async def get_observation_userdata(
    *,
    db: Session = Depends(get_db),
    obs_id: UUID,
) -> Mapping[str, Any]:
    """
    Get userdata json
    """
    obs = db.query(ds.Observation).filter_by(id=obs_id).one()
    return obs.userdata


def _observation_from_db(
    obs: ds.Observation, transformations: Optional[Mapping[str, Any]] = None
) -> js.ObservationOut:
    transformation = None
    if transformations is not None and obs.name in transformations:
        transformation = js.ObservationTransformationOut(
            id=transformations[obs.name].id,
            name=obs.name,
            observation_id=obs.id,
            scale=transformations[obs.name].scale_list,
            active=transformations[obs.name].active_list,
        )
    return js.ObservationOut(
        id=obs.id,
        name=obs.name,
        x_axis=obs.x_axis,
        errors=obs.errors,
        values=obs.values,
        records=[rec.id for rec in obs.records],
        userdata=obs.userdata,
        transformation=transformation,
    )
