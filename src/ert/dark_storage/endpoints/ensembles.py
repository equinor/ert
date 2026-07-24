import io
import logging
from collections import Counter
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query, Response

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_storage, reraise_as_http_errors
from ert.gui.plotting.waterfall_data import compute_waterfall_data
from ert.storage import Storage
from ert.storage.blob_data import BlobStorageData

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
    )


@router.get("/ensembles/{ensemble_id}/blobs")
def get_blobs(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> list[BlobStorageData]:
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        return ensemble.load_blobs()


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


@router.get("/ensembles/{ensemble_id}/waterfall/{parameter_key}")
def get_waterfall_data(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    parameter_key: str,
    nobservations: Annotated[int, Query(ge=1, le=100)] = 10,
) -> Response:
    """Compute waterfall chart data for a scalar parameter update."""
    with reraise_as_http_errors(logger):
        ensemble = storage.get_ensemble(ensemble_id)
        df = compute_waterfall_data(ensemble, unquote(parameter_key), nobservations)

    buf = io.BytesIO()
    df.write_parquet(buf)
    return Response(
        content=buf.getvalue(),
        media_type="application/x-parquet",
    )
