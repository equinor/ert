from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator


class BlobType(StrEnum):
    OBSERVATION_REPORT = "observation_report"
    UPDATE_DATA_TABLE = "update_data_table"
    MATRIX = "matrix"
    SCALING_FACTORS = "scaling_factors"
    RHO_MATRIX = "rho_matrix"
    EVEREST_BATCH_DATA = "everest_batch_data"


class UpdateStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class ObservationReportData(BaseModel):
    blob_type: Literal[BlobType.OBSERVATION_REPORT] = BlobType.OBSERVATION_REPORT
    update_algorithm: str
    summary: dict[str, str] = {}
    status: UpdateStatus = UpdateStatus.COMPLETED
    error_message: str | None = None


class UpdateDataTableData(BaseModel):
    blob_type: Literal[BlobType.UPDATE_DATA_TABLE] = BlobType.UPDATE_DATA_TABLE
    table_name: str
    table_index: int
    summary: dict[str, str] = {}


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
    | UpdateDataTableData
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
        | UpdateDataTableData
        | ScalingFactorsData
        | RhoStorageData
        | EverestBatchData,
        Discriminator("blob_type"),
    ]


@dataclass
class UpdateTable:
    """One tab of an update, as shown while the experiment was running."""

    name: str
    header: list[str]
    rows: Sequence[Sequence[object]]
    summary: dict[str, str] = field(default_factory=dict)
    is_report: bool = False


@dataclass
class StoredUpdate:
    """The update that produced an ensemble, as recorded in storage."""

    update_algorithm: str
    status: UpdateStatus
    tables: list[UpdateTable]
    error_message: str | None = None
