import datetime
import os

import numpy as np
import pytest
from ecl.summary import EclSum

from ert.serialization import _serializer

OBJECTS = [None, {}, {"foo": "bar"}]
OBJECTS_JSON = ["null", "{}", '{"foo": "bar"}']
OBJECTS_YAML = ["null\n...", "{}", "foo: bar"]
assert len(OBJECTS) == len(OBJECTS_JSON) == len(OBJECTS_YAML)


def _create_bin_data(path, length=3):
    """Create synthetic UNSMRY+SMSPEC files in a specified directory"""
    sum_keys = {
        "FOPT": [i for i in range(length)],
        "FOPR": [1] * length,
    }
    dimensions = [10, 10, 10]
    ecl_sum = EclSum.writer("TEST", datetime.date(2000, 1, 1), *dimensions)

    for key in sum_keys:
        ecl_sum.add_variable(key)

    for val, idx in enumerate(range(0, length, 1)):
        t_step = ecl_sum.add_t_step(idx, val)
        for key, item in sum_keys.items():
            t_step[key] = item[idx]

    # libecl can only write UNSMRY+SMSPEC files to current working directory
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        ecl_sum.fwrite()
    finally:
        os.chdir(old_dir)


@pytest.mark.asyncio
async def test_ecl_sum_serializer(tmp_path):
    _create_bin_data(tmp_path, length=3)
    result = await _serializer._ecl_sum_serializer().decode_from_path(
        tmp_path / "TEST", key="FOPT"
    )
    for date, value in result.items():
        assert isinstance(date, str)
        assert isinstance(value, float)
        assert not isinstance(value, np.floating)
    assert result == {
        "2000-01-01 00:00:00": 0,
        "2000-01-02 00:00:00": 1,
        "2000-01-03 00:00:00": 2,
    }


@pytest.mark.asyncio
async def test_ecl_sum_serializer_wrongkey(tmp_path):
    _create_bin_data(tmp_path, length=3)
    with pytest.raises(KeyError):
        await _serializer._ecl_sum_serializer().decode_from_path(
            tmp_path / "TEST", key="BOGUS"
        )


@pytest.mark.parametrize("obj, obj_json", zip(OBJECTS, OBJECTS_JSON))
def test_json_serializer_encode_decode(obj, obj_json):
    json_serializer = _serializer._json_serializer()

    assert json_serializer.encode(obj) == obj_json
    assert json_serializer.decode(obj_json) == obj

    # Test that keyword arguments are passed through
    assert json_serializer.decode(json_serializer.encode(obj, indent=True)) == obj
    assert json_serializer.encode(obj, separators=(",", ":")) == obj_json.replace(
        " ", ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("obj, obj_json", zip(OBJECTS, OBJECTS_JSON))
async def test_json_serializer_path(obj, obj_json, tmp_path):
    json_serializer = _serializer._json_serializer()

    await json_serializer.encode_to_path(obj, tmp_path / "foo.json")
    obj_fromdisk = await json_serializer.decode_from_path(tmp_path / "foo.json")
    assert obj_fromdisk == obj


@pytest.mark.parametrize("obj, obj_yaml", zip(OBJECTS, OBJECTS_YAML))
def test_yaml_serializer(obj, obj_yaml):
    yaml_serializer = _serializer._yaml_serializer()

    assert yaml_serializer.encode(obj).strip() == obj_yaml
    assert yaml_serializer.decode(obj_yaml) == obj

    # Test that keyword arguments are passed through
    assert (
        yaml_serializer.encode(obj, explicit_end=True).strip()
        == obj_yaml.replace("\n...", "") + "\n..."
        # NB: explicit_end is implicitly True on None input
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("obj, obj_yaml", zip(OBJECTS, OBJECTS_YAML))
async def test_yaml_serializer_path(obj, obj_yaml, tmp_path):
    yaml_serializer = _serializer._yaml_serializer()

    await yaml_serializer.encode_to_path(obj, tmp_path / "foo.yaml")
    obj_fromdisk = await yaml_serializer.decode_from_path(tmp_path / "foo.yaml")
    assert obj_fromdisk == obj
