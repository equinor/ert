from uuid import uuid4, UUID
import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
)
from fastapi.responses import Response
from pandas.core.frame import DataFrame
from ert_storage.database import Session, get_db, HAS_AZURE_BLOB_STORAGE
from ert_storage import database_schema as ds

router = APIRouter(tags=["response"])


@router.get("/ensembles/{ensemble_id}/responses/{response_name}/data")
async def get_ensemble_response_dataframe(
    *, db: Session = Depends(get_db), ensemble_id: UUID, response_name: str
) -> Response:
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    records = (
        db.query(ds.Record)
        .filter(ds.Record.realization_index != None)
        .join(ds.RecordInfo)
        .filter_by(
            ensemble_pk=ensemble.pk,
            name=response_name,
            record_class=ds.RecordClass.response,
        )
    ).all()
    df_list = []
    for record in records:
        data_df = pd.DataFrame(record.f64_matrix.content)
        labels = record.f64_matrix.labels
        if labels is not None:
            # if the realization is more than 1D array
            # the next line will produce ValueError exception
            data_df.index = [record.realization_index]
            data_df.columns = labels[0]
        df_list.append(data_df)

    return Response(
        content=pd.concat(df_list, axis=0).to_csv().encode(),
        media_type="text/csv",
    )
