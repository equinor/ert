from typing import Dict

from ert3.data._record import (
    EnsembleRecord,
    InMemoryRecordTransmitter,
    MultiEnsembleRecord,
    Record,
    RecordIndex,
    RecordTransmitter,
    RecordType,
    SharedDiskRecordTransmitter,
    record_data,
)

RealisationsToRecordToTransmitter = Dict[int, Dict[str, RecordTransmitter]]

__all__ = (
    "Record",
    "RecordType",
    "RecordIndex",
    "EnsembleRecord",
    "MultiEnsembleRecord",
    "RecordTransmitter",
    "RecordType",
    "RecordIndex",
    "SharedDiskRecordTransmitter",
    "RealisationsToRecordToTransmitter",
    "InMemoryRecordTransmitter",
    "record_data",
)
