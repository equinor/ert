from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator


class BlobType(StrEnum):
    OBSERVATION_REPORT = "observation_report"
    MATRIX = "matrix"
    SCALING_FACTORS = "scaling_factors"
    RHO_MATRIX = "rho_matrix"
    EVEREST_BATCH_DATA = "everest_batch_data"


class ObservationReportData(BaseModel):
    blob_type: Literal[BlobType.OBSERVATION_REPORT] = BlobType.OBSERVATION_REPORT
    update_algorithm: str


class _MatrixBase(BaseModel):
    update_algorithm: str
    sparse: bool = False
    shape: tuple[int, int] = (0, 0)
    data_type: str
    parameter_group_sizes: dict[str, int] = {}


class MatrixStorageData(_MatrixBase):
    blob_type: Literal[BlobType.MATRIX] = BlobType.MATRIX


class ScalingFactorsData(BaseModel):
    blob_type: Literal[BlobType.SCALING_FACTORS] = BlobType.SCALING_FACTORS
    update_algorithm: str
    num_observations: int
    num_groups: int


class RhoStorageData(_MatrixBase):
    blob_type: Literal[BlobType.RHO_MATRIX] = BlobType.RHO_MATRIX
    param_name: str
    observation_keys: list[str] = []


class EverestBatchData(BaseModel):
    blob_type: Literal[BlobType.EVEREST_BATCH_DATA] = BlobType.EVEREST_BATCH_DATA
    dataframe_name: str


BlobInfo = (
    MatrixStorageData
    | ObservationReportData
    | ScalingFactorsData
    | RhoStorageData
    | EverestBatchData
)


class BlobStorageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    file_size: int
    file_type: str
    name: str
    blob_info: Annotated[
        MatrixStorageData
        | ObservationReportData
        | ScalingFactorsData
        | RhoStorageData
        | EverestBatchData,
        Discriminator("blob_type"),
    ]
