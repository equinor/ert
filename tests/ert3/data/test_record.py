import pydantic
import pytest
import typing

import ert3


@pytest.mark.parametrize(
    "data",
    (
        [1, 2, 3],
        (1.0, 10.0, 42, 999.0),
        [],
        [12.0],
        {1, 2, 3},
        {"a": 0, "b": 1, "c": 2},
        {0: 10, 100: 0},
    ),
)
def test_valid_record(data):
    record = ert3.data.Record(data=data)

    if isinstance(data, set):
        data = tuple(data)

    assert len(data) == len(record.data)
    assert len(data) == len(record.index)
    if isinstance(data, typing.Mapping):
        assert tuple(data.keys()) == record.index
    else:
        assert tuple(range(len(data))) == record.index

    for idx in record.index:
        assert data[idx] == record.data[idx]


@pytest.mark.parametrize(
    "data",
    (
        "a sequence",
        ["one", "two", "three"],
        {"a": "b"},
        {"key": None, "a": None},
    ),
)
def test_invalid_record(data):
    with pytest.raises(pydantic.ValidationError):
        ert3.data.Record(data=data)


@pytest.mark.parametrize(
    ("data", "index"),
    (
        ([1, 2, 3], [0, 1]),  # <- Too short index
        ([1, 2, 3], [0, 1, 5]),  # <- Wrong last index
        ([1, 2, 3], [0, 1, 2, 3]),  # <- Too long index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b", "d")),  # <- Wrong index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b")),  # <- Too short index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b", "c", "d")),  # <- Too long index
    ),
)
def test_inconsistent_index_record(data, index):
    with pytest.raises(pydantic.ValidationError):
        ert3.data.Record(data=data, index=index)


@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
        [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
        [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
    ),
)
def test_valid_ensemble_record(raw_ensrec):
    ensrecord = ert3.data.EnsembleRecord(records=raw_ensrec)

    assert len(raw_ensrec) == len(ensrecord.records) == ensrecord.ensemble_size
    for raw_record, record in zip(raw_ensrec, ensrecord.records):
        raw_record = raw_record["data"]
        assert len(raw_record) == len(record.data)
        for raw_elem, elem in zip(raw_record, record.data):
            assert raw_elem == elem


@pytest.mark.parametrize(
    ("raw_ensrec", "ensemble_size"),
    (
        (
            [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
            2,  # <- Wrong ensemble size
        ),
    ),
)
def test_inconsistent_size_ensemble_record(raw_ensrec, ensemble_size):
    with pytest.raises(pydantic.ValidationError):
        ert3.data.EnsembleRecord(records=raw_ensrec, ensemble_size=ensemble_size)


@pytest.mark.parametrize(
    ("ensemble_size", "raw_multiensrec"),
    (
        (
            3,
            {
                "ens1": {"records": [{"data": [1, 2, 3]} for _ in range(3)]},
                "ens2": {"records": [{"data": {"a": "1", "b": 2}} for _ in range(3)]},
                "ens3": {"records": [{"data": [0]}, {"data": [2]}, {"data": [1]}]},
            },
        ),
        (
            1,
            {
                "ens4": {"records": [{"data": [1, 2, 3]} for _ in range(1)]},
                "ens5": {"records": [{"data": [1, 2, 3]} for _ in range(1)]},
            },
        ),
    ),
)
def test_valid_multi_ensemble_record(ensemble_size, raw_multiensrec):
    multi_ensemblerecord = ert3.data.MultiEnsembleRecord(
        ensemble_records=raw_multiensrec,
    )

    assert len(raw_multiensrec) == len(multi_ensemblerecord)
    assert len(multi_ensemblerecord) == len(multi_ensemblerecord.record_names)
    assert sorted(raw_multiensrec.keys()) == sorted(multi_ensemblerecord.record_names)

    for record_name in multi_ensemblerecord.record_names:
        raw_ensrec = raw_multiensrec[record_name]["records"]
        ensrecord = multi_ensemblerecord.ensemble_records[record_name]
        assert len(raw_ensrec) == len(ensrecord.records) == ensrecord.ensemble_size
        for raw_record, record in zip(raw_ensrec, ensrecord.records):
            raw_record = raw_record["data"]
            assert len(raw_record) == len(record.data)
            for raw_elem, elem in zip(raw_record, record.data):
                assert raw_elem == elem


@pytest.mark.parametrize(
    ("raw_multiensrec", "ensemble_size", "record_names"),
    (
        (
            {
                "ens4": {"records": [{"data": [1, 2, 3]} for _ in range(2)]},
                "ens5": {"records": [{"data": [1, 2, 3]} for _ in range(2)]},
            },
            1,  # <- Wrong ensemble size
            ("ens4", "ens5"),
        ),
        (
            {
                "ens4": {"records": [{"data": [1, 2, 3]} for _ in range(2)]},
                "ens5": {"records": [{"data": [1, 2, 3]} for _ in range(2)]},
            },
            2,
            ("ens4", "ens10"),  # <- Wrong record nmaes
        ),
    ),
)
def test_inconsistent_multi_ensemble_record(
    raw_multiensrec, ensemble_size, record_names
):
    with pytest.raises(pydantic.ValidationError):
        ert3.data.MultiEnsembleRecord(
            ensemble_records=raw_multiensrec,
            ensemble_size=ensemble_size,
            record_names=record_names,
        )
