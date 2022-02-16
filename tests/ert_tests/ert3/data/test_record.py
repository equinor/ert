import json
import typing

import pytest

import ert


@pytest.mark.parametrize(
    "data",
    (
        1,
        1.0,
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

    if record.record_type == ert.data.RecordType.SCALAR_FLOAT:
        assert data == record.data
        assert isinstance(record.data, float)
        assert record.index == ()
    else:
        assert len(data) == len(record.data)
        assert len(data) == len(record.index)

    if isinstance(data, typing.Mapping):
        assert tuple(data.keys()) == record.index
    else:
        if record.record_type != ert.data.RecordType.SCALAR_FLOAT:
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
        {"1": 1, "2": 1.0},
        {"1": 1, 2: 2},
        b"abcde",
        [b"abcde"],
    ),
)
def test_invalid_numerical_record(data):
    with pytest.raises(ert.data.RecordValidationError):
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
    with pytest.raises(ert.data.RecordValidationError):
        ert.data.BlobRecord(data=data)


@pytest.mark.parametrize(
    ("data", "index"),
    (
        ([1, 2, 3], (0, 1)),  # <- Too short index
        ([1, 2, 3], (0, 1, 5)),  # <- Wrong last index
        ([1, 2, 3], (0, 1, 2, 3)),  # <- Too long index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b", "d")),  # <- Wrong index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b")),  # <- Too short index
        ({"a": 0, "b": 1, "c": 2}, ("a", "b", "c", "d")),  # <- Too long index
        ({1: 0, 2: 0}, (1, 2.0)),  # <- Mix of index types
        (1, (0)),  # <- Index should be empty for scalars
    ),
)
def test_inconsistent_index_record(data, index):
    with pytest.raises(ert.data.RecordValidationError):
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
def test_valid_ensemble_record(raw_ensrec, raw_ensrec_to_records, record_type):
    ensrecord = ert.data.RecordCollection(records=raw_ensrec_to_records(raw_ensrec))
    assert ensrecord.record_type == record_type
    assert ensrecord.collection_type != ert.data.RecordCollectionType.UNIFORM
    assert len(raw_ensrec) == len(ensrecord.records) == len(ensrecord)
    for raw_record, record in zip(raw_ensrec, ensrecord.records):
        raw_record = raw_record["data"]
        assert len(raw_record) == len(record.data)
        for raw_elem, elem in zip(raw_record, record.data):
            assert raw_elem == elem


@pytest.mark.parametrize(
    ("raw_ensrec", "record_type"),
    (
        ([{"data": [0.5, 1.1, 2.2]}], ert.data.RecordType.LIST_FLOAT),
        (
            [{"data": {"a": 0.5, "b": 1.1, "c": 2.2}}],
            ert.data.RecordType.MAPPING_STR_FLOAT,
        ),
        ([{"data": {2: 0.5, 5: 1.1, 7: 2.2}}], ert.data.RecordType.MAPPING_INT_FLOAT),
        ([{"data": b"a"}], ert.data.RecordType.BYTES),
    ),
)
def test_valid_uniform_ensemble_record(raw_ensrec, raw_ensrec_to_records, record_type):
    ens_size = 5
    ensrecord = ert.data.RecordCollection(
        records=raw_ensrec_to_records(raw_ensrec),
        length=ens_size,
        collection_type=ert.data.RecordCollectionType.UNIFORM,
    )
    assert ensrecord.record_type == record_type
    assert ensrecord.collection_type == ert.data.RecordCollectionType.UNIFORM
    assert len(ensrecord.records) == len(ensrecord) == ens_size
    raw_record = raw_ensrec[0]["data"]
    assert len(raw_record) == len(ensrecord.records[0].data)
    for raw_elem, elem in zip(raw_record, ensrecord.records[0].data):
        assert raw_elem == elem
    # All records must be references to the same object:
    for record in ensrecord.records[1:]:
        assert record is ensrecord.records[0]


@pytest.mark.parametrize(
    "length, collection_type",
    [
        (None, ert.data.RecordCollectionType.UNIFORM),
        (None, ert.data.RecordCollectionType.NON_UNIFORM),
        (0, ert.data.RecordCollectionType.UNIFORM),
        (0, ert.data.RecordCollectionType.NON_UNIFORM),
        (None, None),
        (0, None),
    ],
)
def test_empty_record_collection(length, collection_type):
    with pytest.raises(ValueError):
        ert.data.RecordCollection(
            records=tuple(), length=length, collection_type=collection_type
        )


@pytest.mark.parametrize(
    ("record_dict"),
    (
        {
            "key_A:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            "key_B:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            "group_OP2": {
                "key_AA:OP2": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
                "key_BA:OP2": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            },
        },
        {
            "key_A:OP2": ert.data.NumericalRecord(data={"a": 0, "b": 1, "c": 2}),
            "key_B:OP2": ert.data.NumericalRecord(
                data={"a": 0, "b": 1, "c": 2},
            ),
            "key_C:OP2": ert.data.NumericalRecord(
                data={"a": 0, "b": 1, "c": 2},
            ),
        },
    ),
)
def test_valid_recordtree_creation(record_dict):
    if isinstance(list(record_dict.values())[0], ert.data.BlobRecord):
        record = ert.data.BlobRecordTree(record_dict=record_dict)
        assert record.record_type == ert.data.RecordType.BLOB_TREE
    else:
        record = ert.data.NumericalRecordTree(record_dict=record_dict)
        assert record.record_type == ert.data.RecordType.NUMERICAL_TREE


@pytest.mark.parametrize(
    ("record_dict"),
    (
        {
            "key_A:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
            "key_B:OP1": ert.data.NumericalRecord(data={"a": 0, "b": 1, "c": 2}),
        },
        {
            "key_B:OP1": ert.data.NumericalRecord(data={"a": 0, "b": 1, "c": 2}),
            "key_A:OP1": ert.data.BlobRecord(data=b"\xF0\x9F\xA6\x89"),
        },
        {
            "key_A:OP2": ert.data.NumericalRecord(data={"a": 0, "b": 1, "c": 2}),
            "key_B:OP2": ert.data.NumericalRecord(
                data=(0, 2, 4),
            ),
            "key_C:OP2": ert.data.NumericalRecord(
                data={"a": 0, "b": 1, "c": 2},
            ),
        },
    ),
)
def test_invalid_recordtree_creation(record_dict):
    with pytest.raises(ert.data.RecordValidationError):
        ert.data.BlobRecordTree(record_dict=record_dict)


def test_invalid_ensemble_record(raw_ensrec_to_records):
    raw_ensrec = [{"data": b"a"}, {"data": [1.1, 2.2]}]
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=raw_ensrec_to_records(raw_ensrec))


def test_uniform_ensemble_record_missing_size(raw_ensrec_to_records):
    with pytest.raises(ValueError):
        ert.data.RecordCollection(
            records=raw_ensrec_to_records([{"data": b"a"}]),
            collection_type=ert.data.RecordCollectionType.UNIFORM,
        )


@pytest.mark.parametrize(
    "raw_ensrec",
    (
        [{"data": {"a": 1.1}}, {"data": [1.1, 2.2]}],
        [{"data": {1: 1.1}}, {"data": [1.1, 2.2]}],
    ),
)
def test_non_uniform_ensemble_record_types(raw_ensrec, raw_ensrec_to_records):
    with pytest.raises(ValueError):
        ert.data.RecordCollection(records=raw_ensrec_to_records(raw_ensrec))


@pytest.mark.parametrize(
    ("raw_ensrec", "ensemble_size"),
    (
        (
            [{"data": [i + 0.5, i + 1.1, i + 2.2]} for i in range(3)],
            2,  # <- Wrong ensemble size
        ),
    ),
)
def test_inconsistent_size_ensemble_record(
    raw_ensrec, raw_ensrec_to_records, ensemble_size
):
    with pytest.raises(ValueError):
        ert.data.RecordCollection(
            records=raw_ensrec_to_records(raw_ensrec), length=ensemble_size
        )


@pytest.mark.asyncio
async def test_load_numeric_record_collection_from_file(designed_coeffs_record_file):
    with open(designed_coeffs_record_file, "r") as f:
        raw_collection = json.load(f)

    transformation = ert.data.SerializationTransformation(
        location=designed_coeffs_record_file,
        mime="application/json",
        direction=ert.data.TransformationDirection.TO_RECORD,
    )
    collection = await ert.data.load_collection_from_file(transformation)
    assert len(collection.records) == len(raw_collection)
    assert len(collection) == len(raw_collection)
    assert collection.record_type != ert.data.RecordType.BYTES
    assert collection.collection_type != ert.data.RecordCollectionType.UNIFORM


@pytest.mark.asyncio
async def test_load_blob_record_collection_from_file(designed_blob_record_file):
    """A single file with random binary content, should be possible to use
    to build a RecordCollection of arbitrary size. For performance reasons,
    each reference in the collection should point to the same object.

    When a single binary file is loaded, a uniform collection is implicit."""
    ens_size = 5
    transformation = ert.data.SerializationTransformation(
        location=designed_blob_record_file,
        mime="application/octet-stream",
        direction=ert.data.TransformationDirection.TO_RECORD,
    )
    collection = await ert.data.load_collection_from_file(
        transformation,
        length=ens_size,
    )
    assert len(collection.records) == ens_size
    assert len(collection) == ens_size
    assert collection.record_type == ert.data.RecordType.BYTES
    assert collection.collection_type == ert.data.RecordCollectionType.UNIFORM
    for record in collection.records[1:]:
        assert record is collection.records[0]
