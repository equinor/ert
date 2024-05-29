from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["ensemble"])
DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> js.EnsembleOut:
    ensemble = storage.get_ensemble(ensemble_id)
    return js.EnsembleOut(
        id=ensemble_id,
        experiment_id=ensemble.experiment_id,
        userdata={"name": ensemble.name, "experiment_name": ensemble.experiment.name},
        size=ensemble.ensemble_size,
    )
