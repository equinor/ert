import json
from ._serializer import Serializer
from typing import Any, TextIO
import ert


class _transmitter_serializer(Serializer):
    _JSON_SCHEMA_VERSION = "0"

    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "_version": self._JSON_SCHEMA_VERSION,
                **obj.to_dict(),
            }
        )

    def decode(self, series: str, *args: Any, **kwargs: Any) -> Any:
        data = json.loads(series, *args, **kwargs)

        if data["_version"] != self._JSON_SCHEMA_VERSION:
            raise RuntimeError(
                f"unexpected transmitter schema version {data['_version']}, expected {self._JSON_SCHEMA_VERSION}"
            )

        del data["_version"]

        type_ = ert.data.RecordTransmitterType(data["transmitter_type"])
        if type_ == ert.data.RecordTransmitterType.in_memory:
            return ert.data.InMemoryRecordTransmitter.from_dict(data)
        elif type_ == ert.data.RecordTransmitterType.shared_disk:
            return ert.data.SharedDiskRecordTransmitter.from_dict(data)
        elif type_ == ert.data.RecordTransmitterType.ert_storage:
            return ert.storage.StorageRecordTransmitter.from_dict(data)
        raise NotImplementedError(f"unsupported transmitter type {type_}")

    def encode_to_file(self, obj: Any, fp: TextIO, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("file IO of record transmitter not yet implemented")

    def decode_from_file(self, fp: TextIO, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("file IO of record transmitter not yet implemented")
