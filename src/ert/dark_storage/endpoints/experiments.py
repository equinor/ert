from uuid import UUID
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm.attributes import flag_modified
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from ert_storage.json_schema.prior import (
    PriorConst,
    PriorTrig,
    PriorNormal,
    PriorLogNormal,
    PriorErtTruncNormal,
    PriorStdNormal,
    PriorUniform,
    PriorErtDUniform,
    PriorLogUniform,
    PriorErtErf,
    PriorErtDErf,
)
from typing import Any, Mapping, List, Type


router = APIRouter(tags=["experiment"])


@router.get("/experiments", response_model=List[js.ExperimentOut])
def get_experiments(
    *,
    db: Session = Depends(get_db),
) -> List[js.ExperimentOut]:
    experiments = db.query(ds.Experiment).all()
    return [_experiment_from_db(exp) for exp in experiments]


@router.get("/experiments/{experiment_id}", response_model=js.ExperimentOut)
def get_experiment_by_id(
    *, db: Session = Depends(get_db), experiment_id: UUID
) -> js.ExperimentOut:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    return _experiment_from_db(experiment)


@router.post("/experiments", response_model=js.ExperimentOut)
def post_experiments(
    *,
    db: Session = Depends(get_db),
    ens_in: js.ExperimentIn,
) -> js.ExperimentOut:
    experiment = ds.Experiment(name=ens_in.name)

    if ens_in.priors:
        db.add_all(
            ds.Prior(
                function=ds.PriorFunction.__members__[prior.function],
                experiment=experiment,
                name=name,
                argument_names=[x[0] for x in prior if isinstance(x[1], (float, int))],
                argument_values=[x[1] for x in prior if isinstance(x[1], (float, int))],
            )
            for name, prior in ens_in.priors.items()
        )

    db.add(experiment)
    db.commit()
    return _experiment_from_db(experiment)


@router.get(
    "/experiments/{experiment_id}/ensembles", response_model=List[js.EnsembleOut]
)
def get_experiment_ensembles(
    *, db: Session = Depends(get_db), experiment_id: UUID
) -> List[ds.Ensemble]:
    return db.query(ds.Ensemble).join(ds.Experiment).filter_by(id=experiment_id).all()


@router.put("/experiments/{experiment_id}/userdata")
async def replace_experiment_userdata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Assign new userdata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    experiment.userdata = body
    db.commit()


@router.patch("/experiments/{experiment_id}/userdata")
async def patch_experiment_userdata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
    body: Any = Body(...),
) -> None:
    """
    Update userdata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    experiment.userdata.update(body)
    flag_modified(experiment, "userdata")
    db.commit()


@router.get("/experiments/{experiment_id}/userdata", response_model=Mapping[str, Any])
async def get_experiment_userdata(
    *,
    db: Session = Depends(get_db),
    experiment_id: UUID,
) -> Mapping[str, Any]:
    """
    Get userdata json
    """
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    return experiment.userdata


@router.delete("/experiments/{experiment_id}")
def delete_experiment(*, db: Session = Depends(get_db), experiment_id: UUID) -> None:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    db.delete(experiment)
    db.commit()


PRIOR_FUNCTION_TO_PYDANTIC: Mapping[ds.PriorFunction, Type[js.Prior]] = {
    ds.PriorFunction.const: PriorConst,
    ds.PriorFunction.trig: PriorTrig,
    ds.PriorFunction.normal: PriorNormal,
    ds.PriorFunction.lognormal: PriorLogNormal,
    ds.PriorFunction.ert_truncnormal: PriorErtTruncNormal,
    ds.PriorFunction.stdnormal: PriorStdNormal,
    ds.PriorFunction.uniform: PriorUniform,
    ds.PriorFunction.ert_duniform: PriorErtDUniform,
    ds.PriorFunction.loguniform: PriorLogUniform,
    ds.PriorFunction.ert_erf: PriorErtErf,
    ds.PriorFunction.ert_derf: PriorErtDErf,
}


def prior_to_dict(prior: ds.Prior) -> dict:
    return (
        PRIOR_FUNCTION_TO_PYDANTIC[prior.function]
        .parse_obj(
            {key: val for key, val in zip(prior.argument_names, prior.argument_values)}
        )
        .dict()
    )


def experiment_priors_to_dict(experiment: ds.Experiment) -> Mapping[str, dict]:
    return {p.name: prior_to_dict(p) for p in experiment.priors}


def _experiment_from_db(exp: ds.Experiment) -> js.ExperimentOut:
    return js.ExperimentOut(
        id=exp.id,
        name=exp.name,
        ensemble_ids=exp.ensemble_ids,
        priors=experiment_priors_to_dict(exp),
        userdata=exp.userdata,
    )
