import pytest

from ert._c_wrappers.enkf.config import ExtParamConfig


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
