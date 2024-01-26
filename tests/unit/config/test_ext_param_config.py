import pytest

from ert.config import ExtParamConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_ext_param_config():
    input_keys = ["key1", "key2", "key3"]
    config = ExtParamConfig("Key", input_keys=input_keys)
    assert len(config) == 3

    for index, (configkey, _) in enumerate(config):
        assert configkey == input_keys[index]

    with pytest.raises(IndexError):
        _ = config[100]

    assert "key1" in config


def test_ext_param_config_suffixes():
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
    config = ExtParamConfig("Key", input_keys=input_dict)

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
        _ = config["no_such_key"]
