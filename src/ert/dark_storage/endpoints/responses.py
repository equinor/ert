from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ert.dark_storage.common import data_for_key
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["response"])

DEFAULT_STORAGE = Depends(get_storage)


@router.get("/ensembles/{ensemble_id}/responses/{response_name}/data")
async def get_ensemble_response_dataframe(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_name: str,
) -> Response:
    dataframe = data_for_key(storage.get_ensemble(ensemble_id), response_name)
    return Response(
        content=dataframe.to_csv().encode(),
        media_type="text/csv",
    )
