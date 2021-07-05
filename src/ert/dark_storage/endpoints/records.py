from uuid import uuid4, UUID
import io
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Mapping, Optional, List, AsyncGenerator
import sqlalchemy as sa
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Header,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm.attributes import flag_modified
from ert_storage.database import Session, get_db
from ert_storage import database_schema as ds, json_schema as js
from ert_storage import exceptions as exc
from ert_storage.endpoints._records_blob import (
    get_blob_handler,
    get_blob_handler_from_record,
    BlobHandler,
)

from fastapi.logger import logger


router = APIRouter(tags=["record"])


class ListRecords(BaseModel):
    ensemble: Mapping[str, str]
    forward_model: Mapping[str, str]


def get_record_by_name(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
) -> ds.Record:
    try:
        return (
            db.query(ds.Record)
            .filter_by(realization_index=realization_index)
            .join(ds.RecordInfo)
            .filter_by(name=name)
            .join(ds.Ensemble)
            .filter_by(id=ensemble_id)
            .one()
        )
    except NoResultFound as e:
        pass

    if realization_index is not None:
        return (
            db.query(ds.Record)
            .filter_by(realization_index=None)
            .join(ds.RecordInfo)
            .filter_by(
                name=name,
                record_type=ds.RecordType.f64_matrix,
            )
            .join(ds.Ensemble)
            .filter_by(id=ensemble_id)
            .one()
        )
    raise exc.NotFoundError(f"Record not found")


def new_record(
    *,
    db: Session = Depends(get_db),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
) -> ds.Record:
    """
    Get ensemble and verify that no record with the name `name` exists
    """
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()

    q = (
        db.query(ds.Record)
        .join(ds.RecordInfo)
        .filter_by(ensemble_pk=ensemble.pk, name=name)
    )
    if realization_index is not None:
        if realization_index not in range(ensemble.size) and ensemble.size != -1:
            raise exc.ExpectationError(
                f"Ensemble '{name}' ('{ensemble_id}') does have a 'size' "
                f"of {ensemble.size}. The posted record is targeting "
                f"'realization_index' {realization_index} which is out "
                f"of bounds."
            )

        q = q.filter(
            (ds.Record.realization_index == None)
            | (ds.Record.realization_index == realization_index)
        )

    if q.count() > 0:
        raise exc.ConflictError(
            f"Ensemble-wide record '{name}' for ensemble '{ensemble_id}' already exists",
        )

    return ds.Record(
        record_info=ds.RecordInfo(
            ensemble=ensemble,
            name=name,
        ),
        realization_index=realization_index,
    )


def new_record_file(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(new_record),
) -> ds.Record:
    record.record_info.record_class = ds.RecordClass.other
    record.record_info.record_type = ds.RecordType.file
    return record


def new_record_matrix(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(new_record),
    prior: Optional[str] = None,
) -> ds.Record:
    ensemble = record.record_info.ensemble
    if record.name in ensemble.parameter_names:
        record_class = ds.RecordClass.parameter
    elif record.name in ensemble.response_names:
        record_class = ds.RecordClass.response
    else:
        record_class = ds.RecordClass.other

    if prior is not None:
        if record_class is not ds.RecordClass.parameter:
            raise exc.UnprocessableError(
                "Priors can only be specified for parameter records"
            )
        record.record_info.prior = db.query(ds.Prior).filter_by(name=prior).one()

    record.record_info.record_class = record_class
    record.record_info.record_type = ds.RecordType.f64_matrix
    return record


@router.post("/ensembles/{ensemble_id}/records/{name}/file")
async def post_ensemble_record_file(
    *,
    db: Session = Depends(get_db),
    bh: BlobHandler = Depends(get_blob_handler),
    record: ds.Record = Depends(new_record_file),
    file: UploadFile = File(...),
) -> None:
    """
    Assign an arbitrary file to the given `name` record.
    """
    record.file = await bh.upload_file(file)
    _create_record(db, record)


@router.put("/ensembles/{ensemble_id}/records/{name}/blob")
async def add_block(
    *,
    db: Session = Depends(get_db),
    bh: BlobHandler = Depends(get_blob_handler),
    record: ds.Record = Depends(get_record_by_name),
    request: Request,
    block_index: int,
) -> None:
    """
    Stage blocks to an existing azure blob record.
    """
    db.add(await bh.stage_blob(record, request, block_index))


@router.post("/ensembles/{ensemble_id}/records/{name}/blob")
async def create_blob(
    *,
    db: Session = Depends(get_db),
    bh: BlobHandler = Depends(get_blob_handler),
    record: ds.Record = Depends(new_record_file),
) -> None:
    """
    Create a record which points to a blob on Azure Blob Storage.
    """
    record.file = bh.create_blob()
    _create_record(db, record)


@router.patch("/ensembles/{ensemble_id}/records/{name}/blob")
async def finalize_blob(
    *,
    db: Session = Depends(get_db),
    bh: BlobHandler = Depends(get_blob_handler),
    record: ds.Record = Depends(get_record_by_name),
) -> None:
    """
    Commit all staged blocks to a blob record
    """
    submitted_blocks = list(
        db.query(ds.FileBlock)
        .filter_by(
            record_name=record.name,
            ensemble_pk=record.ensemble_pk,
            realization_index=record.realization_index,
        )
        .order_by(ds.FileBlock.block_index)
        .all()
    )
    await bh.finalize_blob(submitted_blocks, record)


@router.post(
    "/ensembles/{ensemble_id}/records/{name}/matrix", response_model=js.RecordOut
)
async def post_ensemble_record_matrix(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(new_record_matrix),
    content_type: str = Header("application/json"),
    request: Request,
) -> js.RecordOut:
    """
    Assign an n-dimensional float matrix, encoded in JSON, to the given `name` record.
    """
    if content_type == "application/x-dataframe":
        logger.warning(
            "Content-Type with 'application/x-dataframe' is deprecated. Use 'text/csv' instead."
        )
        content_type = "text/csv"

    labels = None

    try:
        if content_type == "application/json":
            content = np.array(await request.json(), dtype=np.float64)
        elif content_type == "application/x-numpy":
            from numpy.lib.format import read_array

            stream = io.BytesIO(await request.body())
            content = read_array(stream)
        elif content_type == "text/csv":
            stream = io.BytesIO(await request.body())
            df = pd.read_csv(stream, index_col=0, float_precision="round_trip")
            content = df.values
            labels = [
                [str(v) for v in df.columns.values],
                [str(v) for v in df.index.values],
            ]
        else:
            raise ValueError()
    except ValueError:
        if record.realization_index is None:
            message = f"Ensemble-wide record '{record.name}' for needs to be a matrix"
        else:
            message = f"Forward-model record '{record.name}' for realization {record.realization_index} needs to be a matrix"

        raise exc.UnprocessableError(message)

    # Require that the dimensionality of an ensemble-wide parameter matrix is at least 2
    if (
        record.realization_index is None
        and record.record_class is ds.RecordClass.parameter
    ):
        if content.ndim <= 1:
            raise exc.UnprocessableError(
                f"Ensemble-wide parameter record '{record.name}' for ensemble '{record.record_info.ensemble.id}'"
                "must have dimensionality of at least 2"
            )

    matrix_obj = ds.F64Matrix(content=content.tolist(), labels=labels)

    record.f64_matrix = matrix_obj
    return _create_record(db, record)


@router.put("/ensembles/{ensemble_id}/records/{name}/userdata")
async def replace_record_userdata(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(get_record_by_name),
    body: Any = Body(...),
) -> None:
    """
    Assign new userdata json
    """
    record.userdata = body
    db.commit()


@router.patch("/ensembles/{ensemble_id}/records/{name}/userdata")
async def patch_record_userdata(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(get_record_by_name),
    body: Any = Body(...),
) -> None:
    """
    Update userdata json
    """
    record.userdata.update(body)
    flag_modified(record, "userdata")
    db.commit()


@router.get(
    "/ensembles/{ensemble_id}/records/{name}/userdata", response_model=Mapping[str, Any]
)
async def get_record_userdata(
    *,
    record: ds.Record = Depends(get_record_by_name),
) -> Mapping[str, Any]:
    """
    Get userdata json
    """
    return record.userdata


@router.post("/ensembles/{ensemble_id}/records/{name}/observations")
async def post_record_observations(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(get_record_by_name),
    observation_ids: List[UUID] = Body(...),
) -> None:
    observations = (
        db.query(ds.Observation).filter(ds.Observation.id.in_(observation_ids)).all()
    )
    if observations:
        record.observations = observations
        db.commit()
    else:
        raise exc.UnprocessableError(f"Observations {observation_ids} not found!")


@router.get("/ensembles/{ensemble_id}/records/{name}/observations")
async def get_record_observations(
    *,
    db: Session = Depends(get_db),
    record: ds.Record = Depends(get_record_by_name),
) -> List[js.ObservationOut]:
    if record.observations:
        return [
            js.ObservationOut(
                id=obs.id,
                name=obs.name,
                x_axis=obs.x_axis,
                errors=obs.errors,
                values=obs.values,
                records=[rec.id for rec in obs.records],
                userdata=obs.userdata,
            )
            for obs in record.observations
        ]
    return []


GET_CSV_NO_LABELS_DESCRIPTION = """\
If the matrix data contained no label information, the returned CSV uses
integer ranges as the column and row labels.

If the record is a parameter matrix, the rows are the different realizations.
"""


GET_CSV_ALL_LABELS_DESCRIPTION = """\
Returned CSV contains strings for column and row labels. The data itself is numeric.

If the record is a parameter matrix, the rows are the different realizations.
"""


GET_NUMPY_DESCRIPTION = """\
Data encoded with `numpy.lib.format.write_array` and `numpy.lib.format.read_array`.

To parse data using Python, assuming ERT Storage is running on `http://localhost:8000` :

```python
   import io
   import requests
   from numpy.lib.format import read_array

   resp = requests.get(
       "http://localhost:8000/ensembles/{ENSEMBLE_ID}/records/{RECORD_NAME}",
       headers={"Accept": "application/x-numpy"}
   )
   stream = io.BytesIO(resp.content)
   nparray = read_array(stream)

   # Print the numpy array
   print(nparray)
```
"""


@router.get(
    "/ensembles/{ensemble_id}/records/{name}",
    responses={
        200: {
            "description": "Successful fetch of record",
            "content": {
                "application/json": {
                    "examples": {
                        "no-labels": {
                            "value": [[11.5, 12.5, 13.5], [21.5, 22.5, 23.5]],
                        }
                    }
                },
                "text/csv": {
                    "examples": {
                        "no-labels": {
                            "value": ",\t0,\t1,\t2\n"
                            "0,\t11.5,\t12.5,\t13.5\n"
                            "1,\t21.5,\t22.5,\t23.5\n",
                            "description": GET_CSV_NO_LABELS_DESCRIPTION,
                        },
                        "all-labels": {
                            "value": ",\t2010-01-01,\t2015-01-01,\t2020-01-01\n"
                            "real_0,\t11.5,\t\t12.5,\t\t13.5\n"
                            "real_1,\t21.5,\t\t22.5,\t\t23.5\n",
                            "description": GET_CSV_ALL_LABELS_DESCRIPTION,
                        },
                    }
                },
                "application/x-numpy": {
                    "examples": {
                        "success": {
                            "summary": "Fetch data encoded in NPY array format",
                            "description": GET_NUMPY_DESCRIPTION,
                        }
                    }
                },
            },
        }
    },
)
async def get_ensemble_record(
    *,
    db: Session = Depends(get_db),
    bh: BlobHandler = Depends(get_blob_handler),
    record: ds.Record = Depends(get_record_by_name),
    accept: str = Header("application/json"),
    realization_index: Optional[int] = None,
) -> Any:
    """
    Get record with a given `name`. If `realization_index` is not set, look for
    the ensemble-wide record. If it is set, look first for one created by a
    forward-model for the given realization index and then the ensemble-wide
    record.

    Records support multiple data formats. In particular:
    - Matrix:
      Will return n-dimensional float matrix, where n is arbitrary.
    - File:
      Will return the file that was uploaded.
    """
    if accept == "application/x-dataframe":
        logger.warning(
            "Accept with 'application/x-dataframe' is deprecated. Use 'text/csv' instead."
        )
        accept = "text/csv"

    new_realization_index = (
        realization_index if record.realization_index is None else None
    )
    return await _get_record_data(bh, record, accept, new_realization_index)


@router.get("/ensembles/{ensemble_id}/parameters", response_model=List[str])
async def get_ensemble_parameters(
    *, db: Session = Depends(get_db), ensemble_id: UUID
) -> List[str]:
    ensemble = db.query(ds.Ensemble).filter_by(id=ensemble_id).one()
    return ensemble.parameter_names


@router.get(
    "/ensembles/{ensemble_id}/records", response_model=Mapping[str, js.RecordOut]
)
async def get_ensemble_records(
    *, db: Session = Depends(get_db), ensemble_id: UUID
) -> Mapping[str, ds.Record]:
    return {
        rec.name: rec
        for rec in (
            db.query(ds.Record)
            .join(ds.RecordInfo)
            .join(ds.Ensemble)
            .filter_by(id=ensemble_id)
            .all()
        )
    }


@router.get("/records/{record_id}", response_model=js.RecordOut)
async def get_record(*, db: Session = Depends(get_db), record_id: UUID) -> ds.Record:
    return db.query(ds.Record).filter_by(id=record_id).one()


@router.get("/records/{record_id}/data")
async def get_record_data(
    *,
    db: Session = Depends(get_db),
    record_id: UUID,
    accept: Optional[str] = Header(default="application/json"),
) -> Any:
    if accept == "application/x-dataframe":
        logger.warning(
            "Accept with 'application/x-dataframe' is deprecated. Use 'text/csv' instead."
        )
        accept = "text/csv"

    record = db.query(ds.Record).filter_by(id=record_id).one()
    bh = get_blob_handler_from_record(db, record)
    return await _get_record_data(bh, record, accept)


@router.get(
    "/ensembles/{ensemble_id}/responses", response_model=Mapping[str, js.RecordOut]
)
def get_ensemble_responses(
    *, db: Session = Depends(get_db), ensemble_id: UUID
) -> Mapping[str, ds.Record]:
    return {
        rec.name: rec
        for rec in (
            db.query(ds.Record)
            .join(ds.RecordInfo)
            .filter_by(record_class=ds.RecordClass.response)
            .join(ds.Ensemble)
            .filter_by(id=ensemble_id)
            .all()
        )
    }


async def _get_record_data(
    bh: BlobHandler,
    record: ds.Record,
    accept: Optional[str],
    realization_index: Optional[int] = None,
) -> Response:
    type_ = record.record_info.record_type
    if type_ == ds.RecordType.f64_matrix:
        if realization_index is None:
            content = record.f64_matrix.content
        else:
            content = record.f64_matrix.content[realization_index]

        if accept == "application/x-numpy":
            from numpy.lib.format import write_array

            stream = io.BytesIO()
            write_array(stream, np.array(content))

            return Response(
                content=stream.getvalue(),
                media_type="application/x-numpy",
            )
        if accept == "text/csv":
            data = pd.DataFrame(content)
            labels = record.f64_matrix.labels
            if labels is not None and realization_index is None:
                data.columns = labels[0]
                data.index = labels[1]
            elif labels is not None and realization_index is not None:
                # The output is such that rows are realizations. Because
                # `content` is a 1d list in this case, it treats each element as
                # its own row. We transpose the data so that all of the data
                # falls on the same row.
                data = data.T
                data.columns = labels[0]
                data.index = [realization_index]

            return Response(
                content=data.to_csv().encode(),
                media_type="text/csv",
            )
        else:
            return content
    if type_ == ds.RecordType.file:
        return await bh.get_content(record)
    raise NotImplementedError(
        f"Getting record data for type {type_} and Accept header {accept} not implemented"
    )


def _create_record(
    db: Session,
    record: ds.Record,
) -> ds.Record:
    nested = db.begin_nested()
    try:
        db.add(record)
        db.commit()
    except IntegrityError:
        # Assuming this is a UNIQUE constraint failure due to an existing
        # record_info with the same name and ensemble. Try to fetch the
        # record_info
        nested.rollback()
        record_info = record.record_info
        old_record_info = (
            db.query(ds.RecordInfo)
            .filter_by(ensemble=record_info.ensemble, name=record_info.name)
            .one()
        )

        # Check that the parameters match
        if record_info.record_class != old_record_info.record_class:
            raise exc.ConflictError(
                "Record class of new record does not match previous record class",
                new_record_class=record_info.record_class,
                old_record_class=old_record_info.record_class,
            )
        if record_info.record_type != old_record_info.record_type:
            raise exc.ConflictError(
                "Record type of new record does not match previous record type",
                new_record_type=record_info.record_type,
                old_record_type=old_record_info.record_type,
            )

        record = ds.Record(
            record_info=old_record_info,
            f64_matrix=record.f64_matrix,
            file=record.file,
            realization_index=record.realization_index,
        )
        db.add(record)
        db.commit()

    return record
