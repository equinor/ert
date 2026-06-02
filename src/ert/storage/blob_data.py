from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator


class BlobType(StrEnum):
    OBSERVATION_REPORT = "observation_report"
    MATRIX = "matrix"


class ObservationReportData(BaseModel):
    blob_type: Literal[BlobType.OBSERVATION_REPORT] = BlobType.OBSERVATION_REPORT
    update_algorithm: str


class MatrixStorageData(BaseModel):
    blob_type: Literal[BlobType.MATRIX] = BlobType.MATRIX
    update_algorithm: str
    sparse: bool = False
    shape: tuple[int, int] = (0, 0)
    data_type: str


class BlobStorageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    file_size: int
    file_type: str
    name: str
    blob_info: Annotated[
        MatrixStorageData | ObservationReportData,
        Discriminator("blob_type"),
    ]
