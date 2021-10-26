from typing import Dict, List, Optional
import ert
from beartype import beartype


_transmitter_mapping = Dict[int, Dict[str, Optional[ert.data.RecordTransmitter]]]


class _IOMap:
    def __init__(self, iens: List[int], io_names: List[str]) -> None:
        self._matrix: _transmitter_mapping = {
            i: {name: None for name in io_names} for i in iens
        }
        self._validated = False

    def set_transmitter(
        self, iens: int, io_name: str, transmitter: ert.data.RecordTransmitter
    ) -> "_IOMap":
        if self._validated:
            raise RuntimeError("mutating validated iomap")
        self._matrix[iens][io_name] = transmitter
        return self

    def _assert_all_transmitters_specified(self) -> None:
        for real_id, ios in self._matrix.items():
            for name, trans in ios.items():
                if trans is None:
                    raise ValueError(f"'{name}' for real {real_id} was not set")

    def validate(self) -> "_IOMap":
        self._assert_all_transmitters_specified()
        self._validated = True
        return self

    @classmethod
    @beartype
    def from_dict(cls, mapping_dict: _transmitter_mapping) -> "_IOMap":
        mapping = cls([], [])
        mapping._matrix = mapping_dict
        return mapping

    @beartype
    def to_dict(self) -> _transmitter_mapping:
        return self._matrix


class InputMap(_IOMap):
    def validate(self) -> "_IOMap":
        super().validate()
        for real_id, ios in self._matrix.items():
            for name, trans in ios.items():
                if trans and not trans.is_transmitted():
                    raise ValueError(
                        f"input transmitter '{name}' for real {real_id} is not transmitted"
                    )
        return self


class OutputMap(_IOMap):
    def validate(self) -> "_IOMap":
        super().validate()
        for real_id, ios in self._matrix.items():
            for name, trans in ios.items():
                if trans and trans.is_transmitted():
                    raise ValueError(
                        f"output transmitter '{name}' for real {real_id} is transmitted"
                    )
        return self
