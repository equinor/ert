import json

import pytest

from ert._c_wrappers.enkf.config import ExtParamConfig
from ert._c_wrappers.enkf.data import ExtParam


def test_config():
    input_keys = ["key1", "key2", "key3"]
    config = ExtParamConfig("Key", input_keys)
    assert len(config) == 3

    for index, (configkey, _) in enumerate(config):
        assert configkey == input_keys[index]

    with pytest.raises(IndexError):
        _ = config[100]

    keys = []
    for key in config.keys():
        keys.append(key)
    assert keys == input_keys

    assert "key1" in config


def test_config_with_suffixes():
    input_suffixes = [
        ["a", "b", "c"],
        ["2"],
        ["asd", "qwe", "zxc"],
    ]
    input_dict = {
        "key1": input_suffixes[0],
        "key2": input_suffixes[1],
        "key3": input_suffixes[2],
    }
    config = ExtParamConfig("Key", input_dict)

    assert len(config) == 3
    assert "key3" in config
    assert "not_me" not in config
    assert ("key3", "asd") in config
    assert ("key3", "not_me_either") not in config
    assert ("who", "b") not in config

    for configkey, configsuffixes in config:
        assert configkey in input_dict
        assert configsuffixes in input_suffixes

    for k in input_dict:
        configsuffixes = config[k]
        assert configsuffixes in input_suffixes

    with pytest.raises(IndexError):
        _ = config[100]

    with pytest.raises(IndexError):
        _ = config["no_such_key"]

    assert dict(config.items()) == input_dict


def test_data(tmp_path):
    input_keys = ["key1", "key2", "key3"]
    config = ExtParamConfig("Key", input_keys)
    data = ExtParam(config)

    with pytest.raises(IndexError):
        _ = data[100]
    with pytest.raises(IndexError):
        _ = data[-4]

    with pytest.raises(KeyError):
        _ = data["NoSuchKey"]
    with pytest.raises(KeyError):
        _ = data["key1", "a_suffix"]

    assert "key1" in data
    data[0] = 177
    assert data[0] == 177

    data["key2"] = 321
    assert data[-2] == 321

    with pytest.raises(ValueError):
        data.set_vector([1, 2])

    data.set_vector([1, 2, 3])
    for index, value in enumerate(data):
        assert index + 1 == value

    data.export(str(tmp_path / "file.json"))
    with open(tmp_path / "file.json", encoding="utf-8") as filehandle:
        datafromfile = json.load(filehandle)
    for key in data.config.keys():
        assert data[key] == datafromfile[key]


def test_data_with_suffixes():
    input_suffixes = [
        ["a", "b", "c"],
        ["2"],
        ["asd", "qwe", "zxc"],
    ]
    input_dict = {
        "key1": input_suffixes[0],
        "key2": input_suffixes[1],
        "key3": input_suffixes[2],
    }
    config = ExtParamConfig("Key", input_dict)
    data = ExtParam(config)

    with pytest.raises(IndexError):
        _ = data[0]  # Cannot use indices when we have suffixes
    with pytest.raises(TypeError):
        _ = data["key1", 1]
    with pytest.raises(KeyError):
        _ = data["NoSuchKey"]
    with pytest.raises(KeyError):
        _ = data["key1"]  # requires a suffix
    with pytest.raises(KeyError):
        _ = data["key1", "no_such_suffix"]

    data["key1", "a"] = 1
    data["key1", "b"] = 500.5
    data["key2", "2"] = 2.1
    data["key3", "asd"] = -85
    assert data["key1", "a"] == 1
    assert data["key1", "b"] == 500.5
    assert data["key2", "2"] == 2.1
    assert data["key3", "asd"] == -85

    # We don't know what the value is, but it should be possible to read it
    _ = data["key3", "zxc"]
