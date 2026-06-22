import logging
from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Response

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
            "has_func_eval": ensemble.batch_objectives is not None,
            "has_gradient": ensemble.batch_objective_gradient is not None,
        },
        size=ensemble.ensemble_size,
        realization_storage_states=Counter(
            state for states in ensemble.get_ensemble_state() for state in states
        ),
        blobs=ensemble.load_blobs(),
    )


@router.get("/ensembles/{ensemble_id}/blobs/{uri}")
def get_blob(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    uri: str,
) -> Response:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        blob = ensemble.load_blob(uri)

    return Response(
        content=blob,
        media_type="application/octet-stream",
    )
