from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BlobType(StrEnum):
    OBSERVATION_REPORT = "observation_report"
    MATRIX = "matrix"


class BlobStorageData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blob_type: BlobType
    uri: str
    file_size: int
    ensemble_id: str


class MatrixStorageData(BlobStorageData):
    sparse: bool = False
    shape: tuple[int, int] = (0, 0)
