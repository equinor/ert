from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_storage
from ert.shared.storage.extraction import create_priors
from ert.storage import Storage

router = APIRouter(tags=["experiment"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/experiments", response_model=list[js.ExperimentOut])
def get_experiments(
    *,
    storage: Storage = DEFAULT_STORAGE,
) -> list[js.ExperimentOut]:
    return [
        js.ExperimentOut(
            id=experiment.id,
            name=experiment.name,
            ensemble_ids=[ens.id for ens in experiment.ensembles],
            priors=create_priors(experiment),
            userdata={},
            parameters={
                group: [m.model_dump() for m in config.metadata]
                for group, config in experiment.parameter_configuration.items()
            },
            responses={
                response_type: [m.model_dump() for m in config.metadata]
                for response_type, config in experiment.response_configuration.items()
            },
            observations=experiment.response_key_to_observation_key,
        )
        for experiment in storage.experiments
    ]
