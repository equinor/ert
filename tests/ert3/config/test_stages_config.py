import os
import stat
import pathlib
import pydantic
import pytest
from typing import Callable
import ert3


def create_mock_script(path, data=""):
    script_file = pathlib.Path(path)
    script_file.write_text(data)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


@pytest.fixture()
def base_unix_stage_config(tmpdir):
    tmpdir.chdir()
    config = [
        {
            "name": "unix_stage",
            "type": "unix",
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
            "type": "function",
            "input": [{"record": "some_record", "location": "some_location"}],
            "output": [{"record": "some_record", "location": "some_location"}],
            "function": "builtins:sum",
        }
    ]
    yield config


def test_entry_point(base_unix_stage_config):
    config = ert3.config.load_stages_config(base_unix_stage_config)
    config = config[0]
    assert config.name == "unix_stage"
    assert config.script[0] == "poly --help"


@pytest.mark.parametrize(
    "config, expected_error",
    (
        [{"not_a_key": "value"}, "1 validation error"],
        [[{"not_a_key": "value"}], "4 validation errors"],
    ),
)
def test_entry_point_not_valid(config, expected_error):
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        ert3.config.load_stages_config(config)


def test_single_function_step_valid(base_function_stage_config):
    config = ert3.config.load_stages_config(base_function_stage_config)
    assert config[0].name == "function_stage"
    assert isinstance(config[0].function, Callable)
    assert config[0].function.__name__ == "sum"


def test_step_multi_cmd(base_unix_stage_config):
    config = base_unix_stage_config
    config[0]["transportable_commands"].append({"name": "poly2", "location": "poly2"})

    config[0]["script"] = [
        "poly run1",
        "poly2 gogo",
        "poly run2",
        "poly2 abort",
    ]
    create_mock_script(path="poly2")
    config = ert3.config.load_stages_config(config)
    assert config[0].script[0] == "poly run1"
    assert config[0].script[1] == "poly2 gogo"
    assert config[0].script[2] == "poly run2"
    assert config[0].script[3] == "poly2 abort"


def test_step_non_existing_transportable_cmd(base_unix_stage_config):
    invalid_location = "/not/a/file"
    assert not os.path.exists(invalid_location)

    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": invalid_location}
    )

    err_msg = '"/not/a/file" does not exist'
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=err_msg):
        ert3.config.load_stages_config(config)


def test_step_non_executable_transportable_cmd(base_unix_stage_config):
    non_executable = pathlib.Path("an_ordenary_file")
    non_executable.write_text("This is nothing but an ordinary text file")
    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": non_executable}
    )

    err_msg = f"{str(non_executable)} is not executable"
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=err_msg):
        ert3.config.load_stages_config(config)


def test_step_unknown_script(base_unix_stage_config):
    config = base_unix_stage_config
    config[0]["script"].append("unknown_command")

    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"unknown_command is not a known command",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_definition_error(base_function_stage_config):
    config = base_function_stage_config
    config[0]["function"] = "builtinssum"
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Function should be defined as module:function",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_error(base_function_stage_config):
    config = base_function_stage_config
    config[0]["function"] = "builtins:sun"
    with pytest.raises(ImportError):
        ert3.config.load_stages_config(config)


def test_step_unix_and_function_error(base_unix_stage_config):
    config = base_unix_stage_config
    config[0].update({"function": "builtins:sum"})

    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Function defined for unix step",
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
        pydantic.error_wrappers.ValidationError,
        match=r"Scripts defined for a function stage",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_and_command_error(base_function_stage_config):
    config = base_function_stage_config
    config[0].update(
        {"transportable_commands": [{"name": "poly", "location": "poly.py"}]}
    )
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Commands defined for a function stage",
    ):
        ert3.config.load_stages_config(config)


def test_multi_stages_get_script(base_unix_stage_config, base_function_stage_config):
    multi_stage_config = base_unix_stage_config + base_function_stage_config
    config = ert3.config.load_stages_config(multi_stage_config)

    step1 = config.step_from_key("unix_stage")
    assert step1.script == ["poly --help"]
    step2 = config.step_from_key("function_stage")
    assert isinstance(step2.function, Callable)
    assert step2.function.__name__ == "sum"
