from fastapi import APIRouter

from ert_shared.storage.db import Session, Db
from ert_shared.storage import json_schema as js, database_schema as ds

from datetime import datetime

router = APIRouter()


def _ensemble(db: Session, ensemble: ds.Ensemble) -> dict:
    doc = _ensemble_minimal(ensemble)
    doc.update(
        {
            "realizations": [
                {"name": real.index, "index": real.index}
                for real in db.query(ds.Realization)
                .filter_by(ensemble_id=ensemble.id)
                .all()
            ],
            "responses": [
                {"name": resp.name, "id": resp.id}
                for resp in db.query(ds.ResponseDefinition)
                .filter_by(ensemble_id=ensemble.id)
                .all()
            ],
            "parameters": [
                _parameter_minimal(par)
                for par in db.query(ds.Parameter)
                .filter_by(ensemble_id=ensemble.id)
                .all()
            ],
        }
    )
    return doc


def _ensemble_minimal(ensemble: ds.Ensemble) -> dict:
    return {
        "id": ensemble.id,
        "name": ensemble.name,
        "time_created": ensemble.time_created.isoformat(),
        "parent": {
            "id": ensemble.parent.ensemble_reference.id,
            "name": ensemble.parent.ensemble_reference.name,
        }
        if ensemble.parent is not None
        else {},
        "children": [
            {
                "id": child.ensemble_result.id,
                "name": child.ensemble_result.name,
            }
            for child in ensemble.children
        ],
    }


def _parameter_minimal(param: ds.Parameter) -> dict:
    return {
        "id": param.id,
        "key": param.name,
        "group": param.group,
        "prior": {
            "function": param.prior.function,
            "parameter_names": param.prior.parameter_names,
            "parameter_values": param.prior.parameter_values,
        }
        if param.prior is not None
        else {},
    }


@router.get(
    "/ensembles",
)
async def read_items(
    *,
    db: Session = Db(),
):
    return {"ensembles": [_ensemble_minimal(x) for x in db.query(ds.Ensemble).all()]}


@router.get(
    "/ensembles/{id}",
)
async def read_ensemble_by_id(
    *,
    db: Session = Db(),
    id: int,
):
    return _ensemble(db, (db.query(ds.Ensemble).filter_by(id=id).one()))


@router.get("/ensembles/name/{name}")
async def read_ensemble_by_name(
    *,
    db: Session = Db(),
    name: str,
):
    return _ensemble(
        db,
        (
            db.query(ds.Ensemble)
            .filter_by(name=name)
            .order_by(ds.Ensemble.id.desc())
            .first()
        ),
    )


@router.post("/ensembles", response_model=js.Ensemble)
async def create_ensemble(*, db: Session = Db(), ens_in: js.EnsembleCreate):
    update = None
    if ens_in.update is not None:
        prev_ens = (
            db.query(ds.Ensemble.id)
            .filter_by(name=ens_in.update.ensemble_name)
            .order_by(ds.Ensemble.id.desc())
            .first()
        )
        update = ds.Update(
            algorithm=ens_in.update.algorithm, ensemble_reference_id=prev_ens.id
        )

    ens = ds.Ensemble(time_created=datetime.now(), name=ens_in.name, parent=update)
    db.add(ens)

    priors = {
        (p.group, p.key): ds.ParameterPrior(
            group=p.group,
            key=p.key,
            function=p.function,
            parameter_names=p.parameter_names,
            parameter_values=p.parameter_values,
            ensemble=[ens],
        )
        for p in ens_in.priors
    }

    realizations = {
        index: ds.Realization(ensemble=ens, index=index)
        for index in range(ens_in.realizations)
    }
    db.add_all(realizations.values())
    db.add_all(priors.values())

    for index, param in enumerate(ens_in.parameters):
        if len(param.values) != ens_in.realizations:
            raise HTTPException(
                status_code=422,
                detail=f"Length of Ensemble.parameters[{index}].values must be {ens_in.realizations}",
            )

    db.add_all(
        ds.Parameter(
            ensemble=ens,
            name=p.name,
            group=p.group,
            prior=priors.get((p.group, p.name), None),
            values=p.values,
        )
        for p in ens_in.parameters
    )
    db.commit()

    return ens


@router.post("/ensembles/{id}/misfit")
async def create_misfit(
    *,
    db: Session = Db(),
    id: int,
    misfit: js.MisfitCreate,
):
    observation = (
        db.query(ds.Observation.id).filter_by(name=misfit.observation_key).one()
    )

    response_definition = (
        db.query(ds.ResponseDefinition.id)
        .filter_by(ensemble_id=id, name=misfit.response_definition_key)
        .one()
    )

    ensemble = db.query(ds.Ensemble).filter_by(id=id).one()

    obj = ds.ObservationResponseDefinitionLink(
        observation_id=observation.id,
        response_definition_id=response_definition.id,
    )
    if ensemble.parent is not None:
        obj.update_id = ensemble.parent.id

    db.add(obj)
    db.flush()

    realizations = (
        (realization.id, misfit.realizations[realization.index])
        for realization in (
            db.query(ds.Realization)
            .filter_by(ensemble_id=id)
            .filter(ds.Realization.index.in_(misfit.realizations.keys()))
            .all()
        )
    )

    db.add_all(
        ds.Misfit(
            observation_response_definition_link_id=obj.id,
            response_id=realization_id,
            value=value,
        )
        for realization_id, value in realizations
    )
