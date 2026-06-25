from __future__ import annotations

import logging
import os
import uuid as _uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter

logger = logging.getLogger(__name__)


class BlobType(StrEnum):
    OBSERVATION_REPORT = "observation_report"
    MATRIX = "matrix"
    SCALING_FACTORS = "scaling_factors"
    RHO_MATRIX = "rho_matrix"


class ObservationReportData(BaseModel):
    blob_type: Literal[BlobType.OBSERVATION_REPORT] = BlobType.OBSERVATION_REPORT
    update_algorithm: str


class MatrixStorageData(BaseModel):
    blob_type: Literal[BlobType.MATRIX] = BlobType.MATRIX
    update_algorithm: str
    sparse: bool = False
    shape: tuple[int, int] = (0, 0)
    data_type: str
    parameter_group_sizes: dict[str, int] = {}


class ScalingFactorsData(BaseModel):
    blob_type: Literal[BlobType.SCALING_FACTORS] = BlobType.SCALING_FACTORS
    update_algorithm: str
    num_observations: int
    num_groups: int


class RhoStorageData(MatrixStorageData):
    blob_type: Literal[BlobType.RHO_MATRIX] = BlobType.RHO_MATRIX  # type: ignore[assignment]
    param_name: str
    observation_keys: list[str] = []


BlobInfo = (
    MatrixStorageData | ObservationReportData | ScalingFactorsData | RhoStorageData
)


class _StorageWriter(Protocol):
    def _write_transaction(
        self, filename: str | os.PathLike[str], data: bytes
    ) -> None: ...


class BlobStorageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    file_size: int
    file_type: str
    name: str
    blob_info: Annotated[
        MatrixStorageData | ObservationReportData | ScalingFactorsData | RhoStorageData,
        Discriminator("blob_type"),
    ]

    @classmethod
    def save_blob(
        cls,
        *,
        name: str,
        data: bytes,
        blob_info: BlobInfo,
        file_type: str,
        storage: _StorageWriter,
        blob_dir: Path,
    ) -> BlobStorageData:
        """Create a blob record and persist *data* plus its JSON sidecar.

        A random 8-hex-character ID is generated; files are written as
        ``<id>.blob`` and ``<id>.blob.json`` inside *blob_dir*.

        Returns the :class:`BlobStorageData` instance that was persisted.
        """
        blob_id = _uuid.uuid4().hex[:8]
        uri = f"{blob_id}.blob"
        instance = cls(
            uri=uri,
            file_size=len(data),
            file_type=file_type,
            name=name,
            blob_info=blob_info,
        )
        blob_dir.mkdir(parents=True, exist_ok=True)
        storage._write_transaction(blob_dir / uri, data)
        storage._write_transaction(
            blob_dir / f"{uri}.json",
            instance.model_dump_json(indent=2).encode("utf-8"),
        )
        return instance

    @classmethod
    def load_all(
        cls,
        blob_dir: Path,
        blob_type: BlobType | None = None,
    ) -> list[BlobStorageData]:
        """Return metadata for every blob in *blob_dir*, optionally filtered by type."""
        if not blob_dir.exists():
            return []
        adapter: TypeAdapter[BlobStorageData] = TypeAdapter(BlobStorageData)
        results = []
        for json_path in blob_dir.glob("*.json"):
            meta = adapter.validate_json(json_path.read_bytes())
            if blob_type is None or meta.blob_info.blob_type == blob_type:
                results.append(meta)
        return results

    @staticmethod
    def read_bytes(blob_dir: Path, uri: str) -> bytes:
        """Return raw bytes for the blob identified by *uri* inside *blob_dir*.

        Raises
        ------
        FileNotFoundError
            If *uri* escapes *blob_dir* (path traversal) or does not exist.
        """
        resolved_dir = blob_dir.resolve()
        blob_path = (resolved_dir / uri).resolve()
        try:
            blob_path.relative_to(resolved_dir)
        except ValueError:
            logger.warning("Blob URI %s resolves outside of blob directory", uri)
            raise FileNotFoundError(uri) from None
        if not blob_path.exists():
            logger.warning("Blob file %s not found", uri)
            raise FileNotFoundError(uri)
        return blob_path.read_bytes()
