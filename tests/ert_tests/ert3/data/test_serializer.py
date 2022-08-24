import json

import pytest

from ert.serialization import _serializer

OBJECTS = [None, {}, {"foo": "bar"}]
OBJECTS_JSON = ["null", "{}", '{"foo": "bar"}']
OBJECTS_YAML = ["null\n...", "{}", "foo: bar"]
assert len(OBJECTS) == len(OBJECTS_JSON) == len(OBJECTS_YAML)


@pytest.mark.parametrize("obj, obj_json", zip(OBJECTS, OBJECTS_JSON))
def test_json_serializer_encode_decode(obj, obj_json):
    json_serializer = _serializer._json_serializer()

    assert json.loads(json_serializer.encode(obj)) == json.loads(obj_json)
    assert json_serializer.decode(obj_json) == obj

    # Test that indent actually makes a differences:
    if obj:
        assert len(json_serializer.encode(obj, indent=2)) > len(
            json_serializer.encode(obj, indent=1)
        )

    # Test default indent:
    assert len(json_serializer.encode(obj, indent=4)) == len(
        json_serializer.encode(obj)
    )

    # Test ability to reach the most compact json serialization:
    assert json_serializer.encode(
        obj, indent=None, separators=(",", ":")
    ) == obj_json.replace(" ", "")


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
