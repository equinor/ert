import logging

from fastapi import APIRouter, Body, Depends

from ert.config import SurfaceConfig
from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_storage
from ert.storage import Storage

router = APIRouter(tags=["experiment"])
logger = logging.getLogger(__name__)

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
            userdata={},
            parameters={
                group: config.model_dump()
                for group, config in experiment.parameter_configuration.items()
                if not isinstance(config, SurfaceConfig)
            },
            responses={
                response_type: config.model_dump()
                for response_type, config in experiment.response_configuration.items()
            },
            observations=experiment.response_key_to_observation_key,
        )
        for experiment in storage.experiments
    ]
