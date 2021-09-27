import json
import typing

import pydantic
import pytest

import ert


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
def test_valid_numerical_record(data):
    record = ert.data.NumericalRecord(data=data)

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
    (b"", b"abcde"),
)
def test_valid_blob_record(data):
    record = ert.data.BlobRecord(data=data)

    if isinstance(data, set):
        data = tuple(data)

    assert len(data) == len(record.data)


@pytest.mark.parametrize(
    "data",
    (
        "a sequence",
        {"a": "b"},
        {"key": None, "a": None},
        {"1": 1, "2": "2"},
        b"abcde",
        [b"abcde"],
    ),
)
def test_invalid_numerical_record(data):
    with pytest.raises(pydantic.ValidationError):
        ert.data.NumericalRecord(data=data)


@pytest.mark.parametrize(
    "data",
    (
        "a sequence",
        {"a": "b"},
        {"key": None, "a": None},
        {"1": 1, "2": "2"},
        [1, 2, 3],
        {1, 2, 3},
        {"a": 0, "b": 1, "c": 2},
        [b"abcde"],
    ),
)
def test_invalid_blob_record(data):
    with pytest.raises(pydantic.ValidationError):
        ert.data.BlobRecord(data=data)


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
        ert.data.NumericalRecord(data=data, index=index)


@pytest.mark.parametrize(
    ("raw_ensrec", "record_type"),
    (
        (
            [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
            ert.data.RecordType.LIST_FLOAT,
        ),
        (
            [{"data": {"a": i + 0.5, "b": i + 1.1, "c": i + 2.2}} for i in range(5)],
            ert.data.RecordType.MAPPING_STR_FLOAT,
        ),
        (
            [{"data": {2: i + 0.5, 5: i + 1.1, 7: i + 2.2}} for i in range(2)],
            ert.data.RecordType.MAPPING_INT_FLOAT,
        ),
        (
            [{"data": b"a"}, {"data": b"abc"}],
            ert.data.RecordType.BYTES,
        ),
    ),
)
def test_valid_ensemble_record(raw_ensrec, record_type):
    ensrecord = ert.data.RecordCollection(records=raw_ensrec)
    assert ensrecord.record_type == record_type
    assert ensrecord.collection_type != ert.data.RecordCollectionType.UNIFORM
    assert len(raw_ensrec) == len(ensrecord.records) == ensrecord.ensemble_size
    for raw_record, record in zip(raw_ensrec, ensrecord.records):
        raw_record = raw_record["data"]
        assert len(raw_record) == len(record.data)
        for raw_elem, elem in zip(raw_record, record.data):
            assert raw_elem == elem


@pytest.mark.parametrize(
    ("raw_ensrec", "record_type"),
    (
        ({"data": [0.5, 1.1, 2.2]}, ert.data.RecordType.LIST_FLOAT),
        (
            {"data": {"a": 0.5, "b": 1.1, "c": 2.2}},
            ert.data.RecordType.MAPPING_STR_FLOAT,
        ),
        ({"data": {2: 0.5, 5: 1.1, 7: 2.2}}, ert.data.RecordType.MAPPING_INT_FLOAT),
        ({"data": b"a"}, ert.data.RecordType.BYTES),
    ),
)
def test_valid_uniform_ensemble_record(raw_ensrec, record_type):
    ens_size = 5
    ensrecord = ert.data.RecordCollection(
        records=(raw_ensrec,),
        ensemble_size=ens_size,
        collection_type=ert.data.RecordCollectionType.UNIFORM,
    )
    assert ensrecord.record_type == record_type
    assert ensrecord.collection_type == ert.data.RecordCollectionType.UNIFORM
    assert len(ensrecord.records) == ensrecord.ensemble_size == ens_size
    raw_record = raw_ensrec["data"]
    assert len(raw_record) == len(ensrecord.records[0].data)
    for raw_elem, elem in zip(raw_record, ensrecord.records[0].data):
        assert raw_elem == elem
    # All records must be references to the same object:
    for record in ensrecord.records[1:]:
        assert record is ensrecord.records[0]


def test_ensemble_record_not_empty():
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=[])


def test_invalid_ensemble_record():
    raw_ensrec = [{"data": b"a"}, {"data": [1.1, 2.2]}]
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=raw_ensrec)


def test_uniform_ensemble_record_missing_size():
    with pytest.raises(ValueError):
        ert.data.RecordCollection(
            records={"data": b"a"},
            collection_type=ert.data.RecordCollectionType.UNIFORM,
        )


@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": {"a": 1.1}}, {"data": [1.1, 2.2]}],
        [{"data": {1: 1.1}}, {"data": [1.1, 2.2]}],
    ),
)
def test_non_uniform_ensemble_record_types(raw_ensrec):
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=raw_ensrec)


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
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=raw_ensrec, ensemble_size=ensemble_size)


def test_load_numeric_record_collection_from_file(designed_coeffs_record_file):
    with open(designed_coeffs_record_file, "r") as f:
        raw_collection = json.load(f)

    collection = ert.data.load_collection_from_file(
        designed_coeffs_record_file, "application/json"
    )
    assert len(collection.records) == len(raw_collection)
    assert collection.ensemble_size == len(raw_collection)
    assert collection.record_type != ert.data.RecordType.BYTES
    assert collection.collection_type != ert.data.RecordCollectionType.UNIFORM


def test_load_blob_record_collection_from_file(designed_blob_record_file):
    ens_size = 5
    collection = ert.data.load_collection_from_file(
        designed_blob_record_file, "application/octet-stream", ensemble_size=ens_size
    )
    assert len(collection.records) == ens_size
    assert collection.ensemble_size == ens_size
    assert collection.record_type == ert.data.RecordType.BYTES
    assert collection.collection_type == ert.data.RecordCollectionType.UNIFORM
    # All records must be references to the same object:
    for record in collection.records[1:]:
        assert record is collection.records[0]
