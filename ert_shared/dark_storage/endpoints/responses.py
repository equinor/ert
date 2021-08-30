from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ert_shared.dark_storage.enkf import LibresFacade, get_res

router = APIRouter(tags=["response"])


@router.get("/ensembles/{ensemble_id}/responses/{response_name}/data")
async def get_ensemble_response_dataframe(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID, response_name: str
) -> Response:
    raise NotImplementedError
