from pydantic import BaseModel, root_validator
from typing import Mapping, Union, List, Tuple


class _DataElement(BaseModel):
    validate_all = True
    validate_assignment = True
    extra = "forbid"
    allow_mutation = False
    arbitrary_types_allowed = True


def _build_record_index(data):
    if isinstance(data, Mapping):
        return tuple(data.keys())
    else:
        return tuple(range(len(data)))


class Record(_DataElement):
    data: Union[List[float], Mapping[int, float], Mapping[str, float]]
    index: Union[Tuple[int, ...], Tuple[str, ...]]

    def __init__(self, *, data, index=None, **kwargs):
        if index is None:
            index = _build_record_index(data)
        super().__init__(data=data, index=index, **kwargs)

    @root_validator
    def ensure_consistent_index(cls, record):
        assert "data" in record and "index" in record
        norm_record_index = _build_record_index(record["data"])
        assert norm_record_index == record["index"]
        return record


class EnsembleRecord(_DataElement):
    records: Tuple[Record, ...]
    ensemble_size: int

    def __init__(self, *, records, ensemble_size=None, **kwargs):
        if ensemble_size == None:
            ensemble_size = len(records)
        super().__init__(records=records, ensemble_size=ensemble_size, **kwargs)

    @root_validator
    def ensure_consistent_ensemble_size(cls, ensemble_record):
        assert "records" in ensemble_record and "ensemble_size" in ensemble_record
        assert len(ensemble_record["records"]) == ensemble_record["ensemble_size"]
        return ensemble_record


class MultiEnsembleRecord(_DataElement):
    ensemble_records: Mapping[str, EnsembleRecord]
    ensemble_size: int
    record_names: Tuple[str, ...]

    def __init__(
        self, *, ensemble_records, record_names=None, ensemble_size=None, **kwargs
    ):
        if record_names is None:
            record_names = list(ensemble_records.keys())
        if ensemble_size is None:
            first_record = ensemble_records[record_names[0]]
            try:
                ensemble_size = first_record.ensemble_size
            except AttributeError:
                ensemble_size = len(first_record["records"])

        super().__init__(
            ensemble_records=ensemble_records,
            ensemble_size=ensemble_size,
            record_names=record_names,
            **kwargs,
        )

    @root_validator
    def ensure_consistent_ensemble_size(cls, multi_ensemble_record):
        ensemble_size = multi_ensemble_record["ensemble_size"]
        for ensemble_record in multi_ensemble_record["ensemble_records"].values():
            if ensemble_size != ensemble_record.ensemble_size:
                raise AssertionError("Inconsistent ensemble record size")
        return multi_ensemble_record

    @root_validator
    def ensure_consistent_record_names(cls, multi_ensemble_record):
        assert "record_names" in multi_ensemble_record
        record_names = tuple(multi_ensemble_record["ensemble_records"].keys())
        assert multi_ensemble_record["record_names"] == record_names
        return multi_ensemble_record

    def __len__(self):
        return len(self.record_names)
