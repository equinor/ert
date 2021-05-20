from fastapi import APIRouter, Depends
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from uuid import UUID

router = APIRouter(tags=["ensemble"])


@router.post("/updates", response_model=js.UpdateOut)
def create_update(
    *,
    db: Session = Depends(get_db),
    update: js.UpdateIn,
) -> js.UpdateOut:

    ensemble = db.query(ds.Ensemble).filter_by(id=update.ensemble_reference_id).one()
    update_obj = ds.Update(
        algorithm=update.algorithm,
        ensemble_reference_pk=ensemble.pk,
    )
    db.add(update_obj)

    if update.observation_transformations:
        transformations = {t.name: t for t in update.observation_transformations}

        observation_ids = [t.observation_id for t in transformations.values()]
        observations = (
            db.query(ds.Observation)
            .filter(ds.Observation.id.in_(observation_ids))
            .all()
        )

        observation_transformations = [
            ds.ObservationTransformation(
                active_list=transformations[observation.name].active,
                scale_list=transformations[observation.name].scale,
                observation=observation,
                update=update_obj,
            )
            for observation in observations
        ]

        db.add_all(observation_transformations)

    db.commit()
    return _update_from_db(update_obj)


@router.get("/updates/{update_id}", response_model=js.UpdateOut)
def get_update(
    *,
    db: Session = Depends(get_db),
    update_id: UUID,
) -> js.UpdateOut:
    update_obj = db.query(ds.Update).filter_by(id=update_id).one()
    return _update_from_db(update_obj)


def _update_from_db(update_obj: ds.Update) -> js.UpdateOut:
    return js.UpdateOut(
        id=update_obj.id,
        experiment_id=update_obj.ensemble_reference.experiment.id,
        algorithm=update_obj.algorithm,
        ensemble_reference_id=update_obj.ensemble_reference.id
        if update_obj.ensemble_reference is not None
        else None,
        ensemble_result_id=update_obj.ensemble_result.id
        if update_obj.ensemble_result is not None
        else None,
    )
