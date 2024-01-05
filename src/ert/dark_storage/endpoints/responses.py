from typing import Union
from uuid import UUID

from fastapi import APIRouter, Depends, Header, status
from fastapi.responses import Response
from typing_extensions import Annotated

from ert.dark_storage.common import format_dataframe, get_response
from ert.dark_storage.enkf import get_storage
from ert.storage import StorageReader

router = APIRouter(tags=["response"])

DEFAULT_STORAGE = Depends(get_storage)


@router.get(
    "/ensembles/{ensemble_id}/responses/{response_name}/data",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/csv": {},
                "application/x-parquet": {},
            }
        }
    },
)
async def get_ensemble_response_dataframe(
    *,
    storage: StorageReader = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_name: str,
    accept: Annotated[Union[str, None], Header()] = None,
) -> Response:
    dataframe = get_response(storage.get_ensemble(ensemble_id), response_name)
    media_type = accept if accept is not None else "text/csv"
    return format_dataframe(dataframe, media_type)
