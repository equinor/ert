import io
from typing import Optional, List, Type, AsyncGenerator
from uuid import uuid4, UUID

import numpy as np
import pandas as pd
from fastapi import (
    Request,
    UploadFile,
    Depends,
)
from fastapi.logger import logger
from fastapi.responses import Response, StreamingResponse

from ert_storage import database_schema as ds
from ert_storage.database import Session, get_db, HAS_AZURE_BLOB_STORAGE

if HAS_AZURE_BLOB_STORAGE:
    from ert_storage.database import azure_blob_container


class BlobHandler:
    def __init__(
        self,
        db: Session,
        name: Optional[str],
        ensemble_id: Optional[UUID],
        realization_index: Optional[int],
    ):
        self._db = db
        self._name = name
        self._ensemble_id = ensemble_id
        self._realization_index = realization_index

    async def upload_file(
        self,
        file: UploadFile,
    ) -> ds.File:
        return ds.File(
            filename=file.filename,
            mimetype=file.content_type,
            content=await file.read(),
        )

    async def stage_blob(
        self,
        record: ds.Record,
        request: Request,
        block_index: int,
    ) -> ds.FileBlock:
        ensemble = self._db.query(ds.Ensemble).filter_by(id=self._ensemble_id).one()
        block_id = str(uuid4())

        return ds.FileBlock(
            ensemble=ensemble,
            block_id=block_id,
            block_index=block_index,
            record_name=self._name,
            realization_index=self._realization_index,
            content=await request.body(),
        )

    def create_blob(self) -> ds.File:
        return ds.File(
            filename="test",
            mimetype="mime/type",
        )

    async def finalize_blob(
        self, submitted_blocks: List[ds.FileBlock], record: ds.Record
    ) -> None:
        record.file.content = b"".join([block.content for block in submitted_blocks])

    async def get_content(self, record: ds.Record) -> Response:
        assert record.record_type == ds.RecordType.file
        return Response(
            content=record.file.content,
            media_type=record.file.mimetype,
            headers={
                "Content-Disposition": f'attachment; filename="{record.file.filename}"'
            },
        )


class AzureBlobHandler(BlobHandler):
    async def upload_file(
        self,
        file: UploadFile,
    ) -> ds.File:
        key = f"{self._name}@{self._realization_index}@{uuid4()}"
        blob = azure_blob_container.get_blob_client(key)
        await blob.upload_blob(file.file)

        return ds.File(
            filename=file.filename,
            mimetype=file.content_type,
            az_container=azure_blob_container.container_name,
            az_blob=key,
        )

    async def stage_blob(
        self,
        record: ds.Record,
        request: Request,
        block_index: int,
    ) -> ds.FileBlock:
        block_id = str(uuid4())
        blob = azure_blob_container.get_blob_client(record.file.az_blob)
        await blob.stage_block(block_id, await request.body())

        return ds.FileBlock(
            ensemble_pk=record.ensemble_pk,
            block_id=block_id,
            block_index=block_index,
            record_name=self._name,
            realization_index=self._realization_index,
        )

    def create_blob(self) -> ds.File:
        key = f"{self._name}@{self._realization_index}@{uuid4()}"
        blob = azure_blob_container.get_blob_client(key)

        return ds.File(
            filename="test",
            mimetype="mime/type",
            az_container=azure_blob_container.container_name,
            az_blob=key,
        )

    async def finalize_blob(
        self, submitted_blocks: List[ds.FileBlock], record: ds.Record
    ) -> None:
        blob = azure_blob_container.get_blob_client(record.file.az_blob)
        block_ids = [
            block.block_id
            for block in sorted(submitted_blocks, key=lambda x: x.block_index)
        ]
        await blob.commit_block_list(block_ids)

    async def get_content(self, record: ds.Record) -> Response:
        blob = azure_blob_container.get_blob_client(record.file.az_blob)
        download = await blob.download_blob()

        async def chunk_generator() -> AsyncGenerator[bytes, None]:
            async for chunk in download.chunks():
                yield chunk

        return StreamingResponse(
            chunk_generator(),
            media_type=record.file.mimetype,
            headers={
                "Content-Disposition": f'attachment; filename="{record.file.filename}"'
            },
        )


def get_blob_handler(
    *,
    db: Session = Depends(get_db),
    name: str,
    ensemble_id: UUID,
    realization_index: Optional[int] = None,
) -> BlobHandler:
    blob_handler: Type[BlobHandler]
    if HAS_AZURE_BLOB_STORAGE:
        blob_handler = AzureBlobHandler
    else:
        blob_handler = BlobHandler
    return blob_handler(
        db=db, name=name, ensemble_id=ensemble_id, realization_index=realization_index
    )


def get_blob_handler_from_record(db: Session, record: ds.Record) -> BlobHandler:
    blob_handler: Type[BlobHandler]
    if HAS_AZURE_BLOB_STORAGE:
        blob_handler = AzureBlobHandler
    else:
        blob_handler = BlobHandler
    return blob_handler(
        db=db,
        name=record.name,
        ensemble_id=record.record_info.ensemble.id,
        realization_index=record.realization_index,
    )
