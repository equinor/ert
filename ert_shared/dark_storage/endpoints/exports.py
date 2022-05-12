import pandas as pd
from uuid import UUID
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response
from ert_shared.dark_storage.enkf import LibresFacade, get_res, get_name
from ert_shared.dark_storage.common import export_eclipse_summary_data

router = APIRouter(tags=["exports"])

@router.get(
    "/ensembles/{ensemble_id}/export/csv",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/csv": {}},
            "description": "Exports emsemble responses as csv",
        }
    },
)
async def get_eclipse_summary_data(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    column_keys: Optional[List[str]] = Query(None),
    time_index: Optional[Any] = None,
) -> Response:
    ensemble_name : str = get_name("ensemble", ensemble_id)
    return Response(
        content=export_eclipse_summary_data(res, ensemble_name, time_index, column_keys).to_csv(index=True),
        media_type="text/csv",
    )
