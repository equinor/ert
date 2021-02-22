from fastapi import APIRouter, HTTPException

from ert_shared.storage.db import Session, Db
from ert_shared.storage import json_schema as js, database_schema as ds

from datetime import datetime

router = APIRouter()
import logging

logger = logging.getLogger()


def _ensemble(db: Session, ensemble: ds.Ensemble) -> dict:
    doc = _ensemble_minimal(ensemble)
    doc.update(
        {
            "realizations": ensemble.num_realizations,
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

    ens = ds.Ensemble(
        time_created=datetime.now(),
        name=ens_in.name,
        num_realizations=ens_in.realizations,
    )
    db.add(ens)

    if ens_in.update_id is not None:
        update_obj = db.query(ds.Update).get(ens_in.update_id)
        update_obj.ensemble_result = ens

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

    db.add_all(priors.values())

    for index, param in enumerate(ens_in.parameters):
        if len(param.values) != ens_in.realizations:
            raise HTTPException(
                status_code=422,
                detail=f"Length of Ensemble.parameters[{index}].values must be {ens_in.realizations}, got {len(param.values)}",
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

    existing_observations = {obs.name: obs for obs in db.query(ds.Observation).all()}

    # Adding only new observations. If observations exist, we link to the.
    # existing ones. This will fail if users change the values in observation
    # config file
    observations = {}
    to_be_created = []
    for obs in ens_in.observations:
        try:
            observations[obs.name] = existing_observations[obs.name]
        except KeyError:
            new_obs = ds.Observation(
                name=obs.name,
                x_axis=obs.x_axis,
                values=obs.values,
                errors=obs.errors,
            )
            observations[obs.name] = new_obs
            to_be_created.append(new_obs)
    db.add_all(to_be_created)

    response_definitions = {
        response_name: ds.ResponseDefinition(name=response_name, ensemble=ens)
        for response_name in ens_in.response_observation_link
    }
    db.add_all(response_definitions.values())

    db.add_all(
        ds.ObservationResponseDefinitionLink(
            observation=observations[observation_name],
            response_definition=response_definitions[response_name],
        )
        for response_name, observation_name in ens_in.response_observation_link.items()
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

    obj = (
        db.query(ds.ObservationResponseDefinitionLink)
        .filter_by(
            observation_id=observation.id, response_definition_id=response_definition.id
        )
        .one()
    )

    if ensemble.parent is not None:
        obj.update_id = ensemble.parent.id

    obj = (
        db.query(ds.ObservationResponseDefinitionLink)
        .filter_by(
            observation_id=observation.id, response_definition_id=response_definition.id
        )
        .one()
    )
    db.flush()

    responses = (
        (response, misfit.realizations[response.index])
        for response in (
            db.query(ds.Response)
            .filter(ds.Response.index.in_(misfit.realizations.keys()))
            .join(ds.Response.response_definition)
            .filter(ds.ResponseDefinition.ensemble_id == id)
            .all()
        )
    )

    db.add_all(
        ds.Misfit(
            observation_response_definition_link=obj,
            response=response,
            value=value,
        )
        for response, value in responses
    )


@router.post("/ensembles/{id}/updates", response_model=js.Update)
async def create_observation_transformation(
    *,
    db: Session = Db(),
    id: int,
    update: js.UpdateCreate,
):
    try:
        update_obj = ds.Update(
            algorithm=update.algorithm,
            ensemble_reference_id=update.ensemble_reference_id,
        )
        db.add(update_obj)
        db.commit()

        observations = [
            db.query(ds.Observation)
            .filter_by(name=observation_transformation.name)
            .one()
            for observation_transformation in update.observation_transformations
        ]

        observation_transformations = [
            ds.ObservationTransformation(
                active_list=observation_transformation.active,
                scale_list=observation_transformation.scale,
                observation=observation,
                update=update_obj,
            )
            for observation_transformation, observation in zip(
                update.observation_transformations, observations
            )
        ]

        db.add_all(observation_transformations)
        return update_obj
    except Exception as e:
        logger.error(e)
        raise
