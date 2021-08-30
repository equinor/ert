from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from ert_shared.dark_storage.enkf import LibresFacade, get_res


router = APIRouter(tags=["misfits"])


@router.get(
    "/compute/misfits",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/csv": {}},
        }
    },
)
async def get_response_misfits(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    response_name: str,
    realization_index: Optional[int] = None,
    summary_misfits: bool = False,
) -> Response:
    raise NotImplementedError
