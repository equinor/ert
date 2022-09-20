from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ert.dark_storage.common import data_for_key
from ert.dark_storage.enkf import LibresFacade, get_name, get_res

router = APIRouter(tags=["response"])


@router.get("/ensembles/{ensemble_id}/responses/{response_name}/data")
async def get_ensemble_response_dataframe(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID, response_name: str
) -> Response:
    ensemble_name = get_name("ensemble", ensemble_id)
    dataframe = data_for_key(res, ensemble_name, response_name)
    return Response(
        content=dataframe.to_csv().encode(),
        media_type="text/csv",
    )
