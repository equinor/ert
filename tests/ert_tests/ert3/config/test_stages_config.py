import os
import pathlib
import stat
from typing import Callable
from unittest.mock import patch

import pytest

import ert3
import ert


def create_mock_script(path, data=""):
    script_file = pathlib.Path(path)
    script_file.write_text(data)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


@pytest.fixture()
def base_unix_stage_config(tmpdir):
    with tmpdir.as_cwd():
        config = [
            {
                "name": "unix_stage",
                "input": [{"record": "some_record", "location": "some_location"}],
                "output": [{"record": "some_record", "location": "some_location"}],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": ["poly --help"],
            }
        ]
        create_mock_script("poly.py")
        yield config


@pytest.fixture()
def base_function_stage_config(tmpdir):
    config = [
        {
            "name": "function_stage",
            "input": [{"record": "some_record", "location": "some_location"}],
            "output": [{"record": "some_record", "location": "some_location"}],
            "function": "builtins:sum",
        }
    ]
    yield config


@pytest.fixture(params=["base_unix_stage_config", "base_function_stage_config"])
def base_stage_config(request):
    yield request.getfixturevalue(request.param)


def test_entry_point(base_unix_stage_config):
    config = ert3.config.load_stages_config(base_unix_stage_config)
    config = config[0]
    assert config.name == "unix_stage"
    assert config.script[0] == "poly --help"


@pytest.mark.parametrize(
    "config, expected_error",
    (
        [{"not_a_key": "value"}, "1 validation error"],
        [[{"not_a_key": "value"}], "11 validation errors"],
    ),
)
def test_entry_point_not_valid(config, expected_error):
    with pytest.raises(ert.exceptions.ConfigValidationError, match=expected_error):
        ert3.config.load_stages_config(config)


def test_check_loaded_mime_types_new_default(base_unix_stage_config):
    """If a Record's default mime is set to something other than an empty string,
    the `_ensure_mime` validator will not modify the mime,
    which may lead to an invalid mime.
    """
    from ert3.config._stages_config import StageIO

    raw_config = base_unix_stage_config

    with patch.object(
        StageIO.__dict__["__fields__"]["mime"], "default", "application/not_a_valid_mime"
    ):
        with pytest.raises(ert.exceptions.ConfigValidationError):
            _ = ert3.config.load_stages_config(raw_config)


def test_check_loaded_mime_types(base_unix_stage_config):
    """Test mimetype in transportable commands, input, and output"""
    raw_config = base_unix_stage_config
    raw_config[0]["input"].append(
        {"record": "some_json_record", "location": "some_location.json"}
    )
    raw_config[0]["output"].append(
        {"record": "some_json_record", "location": "some_location.json"}
    )
    config = ert3.config.load_stages_config(raw_config)
    # Check transportable_commands
    assert (
        config[0].transportable_commands[0].mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    )
    # Check input
    assert config[0].input[0].mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    assert config[0].input["some_json_record"].mime == "application/json"
    # Check output
    assert config[0].output[0].mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    assert config[0].output["some_json_record"].mime == "application/json"


def test_step_multi_cmd(base_unix_stage_config):
    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "poly2", "location": "poly2", "mime": "text/x-python"}
    )

    config[0]["script"] = [
        "poly run1",
        "poly2 gogo",
        "poly run2",
        "poly2 abort",
    ]
    create_mock_script(path="poly2")

    with pytest.raises(ert.exceptions.ConfigValidationError):
        config = ert3.config.load_stages_config(config)


def test_step_non_existing_transportable_cmd(base_unix_stage_config):
    invalid_location = "/not/a/file"
    assert not os.path.exists(invalid_location)

    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": invalid_location}
    )

    err_msg = '"/not/a/file" does not exist'
    with pytest.raises(ert.exceptions.ConfigValidationError, match=err_msg):
        ert3.config.load_stages_config(config)


def test_step_function_definition_error(base_function_stage_config):
    config = base_function_stage_config
    config[0]["function"] = "builtinssum"
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"Function should be defined as module:function",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_error(base_function_stage_config):
    config = base_function_stage_config
    config[0]["function"] = "builtins:sun"
    with pytest.raises(ImportError):
        ert3.config.load_stages_config(config)


def test_step_unix_and_function(base_unix_stage_config):
    config = base_unix_stage_config
    config[0].update({"function": "builtins:sum"})
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match=r"extra fields not permitted"
    ):
        ert3.config.load_stages_config(config)


def test_step_function_module_error(base_function_stage_config):
    config = base_function_stage_config
    config[0]["function"] = "builtinx:sum"
    with pytest.raises(
        ModuleNotFoundError,
        match=r"No module named 'builtinx'",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_and_script_error(base_function_stage_config):
    config = base_function_stage_config
    config[0].update({"script": ["poly --help"]})
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"extra fields not permitted",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_and_command_error(base_function_stage_config):
    config = base_function_stage_config
    config[0].update(
        {"transportable_commands": [{"name": "poly", "location": "poly.py"}]}
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"extra fields not permitted",
    ):
        ert3.config.load_stages_config(config)


def test_multi_stages_get_script(base_unix_stage_config, base_function_stage_config):
    multi_stage_config = base_unix_stage_config + base_function_stage_config
    config = ert3.config.load_stages_config(multi_stage_config)

    step1 = config.step_from_key("unix_stage")
    assert [cmd for cmd in step1.script] == ["poly --help"]
    step2 = config.step_from_key("function_stage")
    assert isinstance(step2.function, Callable)
    assert step2.function.__name__ == "sum"


def test_base_immutable(base_stage_config):
    config = ert3.config.load_stages_config(base_stage_config)

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0] = config[0]


def test_stage_immutable(base_stage_config):
    config = ert3.config.load_stages_config(base_stage_config)

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].name = "new-name"


def test_stage_unknown_field(base_stage_config):
    base_stage_config[0]["unknown"] = "field"
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(base_stage_config)


def test_stage_input_immutable(base_stage_config):
    config = ert3.config.load_stages_config(base_stage_config)
    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].input[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].input[0].record = None


def test_stage_input_unknown_field(base_stage_config):
    base_stage_config[0]["input"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(base_stage_config)


def test_stage_output_immutable(base_stage_config):
    config = ert3.config.load_stages_config(base_stage_config)
    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].output[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].output[0].record = None


def test_stage_output_unknown_field(base_stage_config):
    base_stage_config[0]["output"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(base_stage_config)


def test_stage_transportable_command_immutable(base_unix_stage_config):
    config = ert3.config.load_stages_config(base_unix_stage_config)

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].transportable_commands[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].transportable_commands[0].name = "some-name"


def test_stage_transportable_command_unknown_field(base_unix_stage_config):
    base_unix_stage_config[0]["transportable_commands"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(base_unix_stage_config)


def test_stage_script_immutable(base_unix_stage_config):
    config = ert3.config.load_stages_config(base_unix_stage_config)

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].script[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].script[0][0] = "x"


def test_indexed_ordered_dict():
    idx_ordered_dict = ert3.config.IndexedOrderedDict(
        {"record_key_1": "record_value_1", "record_key_2": "record_value_2"}
    )
    assert idx_ordered_dict[0] == idx_ordered_dict["record_key_1"]
