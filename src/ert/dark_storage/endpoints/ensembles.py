import logging
from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_storage, reraise_as_http_errors
from ert.storage import Storage

router = APIRouter(tags=["ensemble"])
logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> js.EnsembleOut:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)

    return js.EnsembleOut(
        id=ensemble_id,
        experiment_id=ensemble.experiment_id,
        userdata={
            "name": ensemble.name,
            "experiment_name": ensemble.experiment.name,
            "started_at": ensemble.started_at,
        },
        size=ensemble.ensemble_size,
        realization_storage_states=Counter(
            state for states in ensemble.get_ensemble_state() for state in states
        ),
    )
