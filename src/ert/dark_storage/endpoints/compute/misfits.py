from ert_storage.database_schema.record import F64Matrix
import numpy as np
import pandas as pd
from uuid import UUID
from typing import Any, Optional, List
import sqlalchemy as sa
from fastapi.responses import Response
from fastapi import APIRouter, Depends, status
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from ert_storage import exceptions as exc
from ert_storage.compute import calculate_misfits_from_pandas, misfits

router = APIRouter(tags=["misfits"])


@router.get(
    "/compute/misfits",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/csv": {}},
            "description": "Return misfits as csv, where columns are realizations.",
        }
    },
)
async def get_response_misfits(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
    response_name: str,
    realization_index: Optional[int] = None,
    summary_misfits: bool = False,
) -> Response:
    """
    Compute univariate misfits for response(s)
    """

    response_query = (
        db.query(ds.Record)
        .filter(ds.Record.observations != None)
        .join(ds.RecordInfo)
        .filter_by(
            name=response_name,
            record_type=ds.RecordType.f64_matrix,
        )
        .join(ds.Ensemble)
        .filter_by(id=ensemble_id)
    )
    if realization_index is not None:
        responses = [
            response_query.filter(
                ds.Record.realization_index == realization_index
            ).one()
        ]
    else:
        responses = response_query.all()

    observation_df = None
    response_dict = {}
    for response in responses:
        data_df = pd.DataFrame(response.f64_matrix.content)
        labels = response.f64_matrix.labels
        if labels is not None:
            data_df.columns = labels[0]
            data_df.index = labels[1]
        response_dict[response.realization_index] = data_df
        if observation_df is None:
            # currently we expect only a single observation object, while
            # later in the future this might change
            obs = response.observations[0]
            observation_df = pd.DataFrame(
                data={"values": obs.values, "errors": obs.errors}, index=obs.x_axis
            )

    try:
        result_df = calculate_misfits_from_pandas(
            response_dict, observation_df, summary_misfits
        )
    except Exception as misfits_exc:
        raise exc.UnprocessableError(f"Unable to compute misfits: {misfits_exc}")
    return Response(
        content=result_df.to_csv().encode(),
        media_type="text/csv",
    )
