from uuid import UUID

from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm.attributes import flag_modified
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from typing import Any, Mapping
from ert_storage import exceptions as exc

router = APIRouter(tags=["ensemble"])


@router.post("/experiments/{experiment_id}/ensembles", response_model=js.EnsembleOut)
def post_ensemble(
    *, db: Session = Depends(get_db), ens_in: js.EnsembleIn, experiment_id: UUID
) -> ds.Ensemble:

    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    active_reals = (
        ens_in.active_realizations
        if ens_in.active_realizations
        else list(range(ens_in.size))
    )

    if ens_in.size > 0:
        if max(active_reals) > ens_in.size - 1:
            raise exc.ExpectationError(
                f"Ensemble active realization index {max(active_reals)} out of realization range [0,{ ens_in.size - 1}]"
            )
        if len(set(active_reals)) != len(active_reals):
            raise exc.ExpectationError(
                f"Non unique active realization index list not allowed {active_reals}"
            )

    ens = ds.Ensemble(
        parameter_names=ens_in.parameter_names,
        response_names=ens_in.response_names,
        experiment=experiment,
        size=ens_in.size,
        userdata=ens_in.userdata,
        active_realizations=active_reals,
    )
    db.add(ens)

    if ens_in.update_id:
        update_obj = db.query(ds.Update).filter_by(id=ens_in.update_id).one()
        update_obj.ensemble_result = ens
    db.commit()

    return ens


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(*, db: Session = Depends(get_db), ensemble_id: UUID) -> ds.Ensemble:
    return db.query(ds.Ensemble).filter_by(id=ensemble_id).one()


@router.put("/ensembles/{ensemble_id}/userdata")
async def replace_ensemble_userdata(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Assign new userdata json
    """
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    ensemble.userdata = body
    db.commit()


@router.patch("/ensembles/{ensemble_id}/userdata")
async def patch_ensemble_userdata(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Update userdata json
    """
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    ensemble.userdata.update(body)
    flag_modified(ensemble, "userdata")
    db.commit()


@router.get("/ensembles/{ensemble_id}/userdata", response_model=Mapping[str, Any])
async def get_ensemble_userdata(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
) -> Mapping[str, Any]:
    """
    Get userdata json
    """
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    return ensemble.userdata
