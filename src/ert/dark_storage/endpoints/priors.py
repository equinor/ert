from uuid import UUID
from typing import Mapping, Type
from fastapi import APIRouter, Depends
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

router = APIRouter(tags=["prior"])


@router.get("/experiments/{experiment_id}/priors")
def get_priors(
    *, db: Session = Depends(get_db), experiment_id: UUID
) -> Mapping[str, dict]:
    experiment = db.query(ds.Experiment).filter_by(id=experiment_id).one()
    return experiment_priors_to_dict(experiment)


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
