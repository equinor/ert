import os
import pathlib
import pydantic
import pytest
import shutil
from typing import Callable
import ert3


_EXAMPLES_ROOT = (
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "examples"
)
_POLY_WORKSPACE_NAME = "polynomial"
_POLY_WORKSPACE = _EXAMPLES_ROOT / _POLY_WORKSPACE_NAME
_POLY_EXEC = _POLY_WORKSPACE / "poly.py"


def _example_config():
    return [
        {
            "name": "evaluate_polynomial",
            "type": "unix",
            "input": [{"record": "coefficients", "location": "coefficients.json"}],
            "output": [{"record": "polynomial_output", "location": "output.json"}],
            "transportable_commands": [{"name": "poly", "location": "poly.py"}],
            "script": ["poly --coefficients coefficients.json --output output.json"],
        }
    ]


def test_entry_point(tmpdir):
    tmpdir.chdir()
    shutil.copy2(_POLY_EXEC, "poly.py")
    config = ert3.config.load_stages_config(_example_config())
    config = config[0]
    assert config.name == "evaluate_polynomial"


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


def test_step_valid():
    config = ert3.config.load_stages_config(
        [
            {
                "name": "some_name",
                "type": "unix",
                "input": [{"record": "some_record", "location": "some_location"}],
                "output": [{"record": "some_record", "location": "some_location"}],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": ["poly --help"],
            }
        ]
    )
    assert config[0].name == "some_name"
    assert config[0].script[0] == "poly --help"


def test_single_function_step_valid():
    config = ert3.config.load_stages_config(
        [
            {
                "name": "minimal_function_stage",
                "type": "function",
                "input": [{"record": "some_record", "location": "some_location"}],
                "output": [{"record": "some_record", "location": "some_location"}],
                "function": "builtins:sum",
            }
        ]
    )
    assert config[0].name == "minimal_function_stage"
    assert isinstance(config[0].function, Callable)
    assert config[0].function.__name__ == "sum"


def test_step_multi_cmd(tmpdir):
    tmpdir.chdir()
    shutil.copy2(_POLY_EXEC, "poly.py")
    shutil.copy2(_POLY_EXEC, "poly2")

    config = _example_config()
    config[0]["transportable_commands"].append({"name": "poly2", "location": "poly2"})

    config[0]["script"] = [
        "poly run1",
        "poly2 gogo",
        "poly run2",
        "poly2 abort",
    ]
    ert3.config.load_stages_config(config)


def test_step_non_existing_transportable_cmd(tmpdir):
    tmpdir.chdir()
    shutil.copy2(_POLY_EXEC, "poly.py")

    invalid_location = "/not/a/file"
    assert not os.path.exists(invalid_location)

    config = _example_config()
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": invalid_location}
    )

    err_msg = '"/not/a/file" does not exist'
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=err_msg):
        ert3.config.load_stages_config(config)


def test_step_non_executable_transportable_cmd(tmpdir):
    tmpdir.chdir()
    shutil.copy2(_POLY_EXEC, "poly.py")

    non_executable = "an_ordenary_file"
    with open(non_executable, "w") as f:
        f.write("This is nothing but an ordinary text file")

    config = _example_config()
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": non_executable}
    )

    err_msg = "an_ordenary_file is not executable"
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=err_msg):
        ert3.config.load_stages_config(config)


def test_step_unknown_script(tmpdir):
    tmpdir.chdir()
    shutil.copy2(_POLY_EXEC, "poly.py")

    config = _example_config()
    config[0]["script"].append("unknown_command")

    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"unknown_command is not a known command",
    ):
        ert3.config.load_stages_config(config)


def test_step_function_definition_error():
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Function should be defined as module:function",
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "function",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "function": "builtinssum",
                }
            ]
        )


def test_step_function_error():
    with pytest.raises(
        ImportError,
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "function",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "function": "builtins:sun",
                }
            ]
        )


def test_step_unix_and_function_error():
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Function defined for unix step",
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "unix",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "input": [{"record": "some_record", "location": "some_location"}],
                    "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                    "script": ["poly --help"],
                    "function": "builtins:sum",
                }
            ]
        )


def test_step_function_module_error():
    with pytest.raises(
        ModuleNotFoundError,
        match=r"No module named 'builtinx'",
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "function",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "function": "builtinx:sum",
                }
            ]
        )


def test_step_function_and_script_error():
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Scripts defined for a function stage",
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "function",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "function": "builtins:sum",
                    "script": [
                        "poly --coefficients coefficients.json --output output.json"
                    ],
                }
            ]
        )


def test_step_function_and_command_error():
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=r"Commands defined for a function stage",
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "minimal_function_stage",
                    "type": "function",
                    "output": [{"record": "some_record", "location": "some_location"}],
                    "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                    "function": "builtins:sum",
                }
            ]
        )


def test_mutli_stages_get_script():
    config = ert3.config.load_stages_config(
        [
            {
                "name": "stage_1",
                "type": "unix",
                "input": [{"record": "some_record", "location": "some_file"}],
                "output": [{"record": "some_record", "location": "some_file"}],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": [
                    "poly --coefficients coefficients.json --output output.json"
                ],
            },
            {
                "name": "stage_2",
                "type": "function",
                "input": [{"record": "some_record", "location": "some_file"}],
                "output": [{"record": "fun_record", "location": "fun_file"}],
                "function": "builtins:sum",
            },
        ]
    )
    step1 = config.step_from_key("stage_1")
    assert step1.script == [
        "poly --coefficients coefficients.json --output output.json"
    ]
    step2 = config.step_from_key("stage_2")
    assert isinstance(step2.function, Callable)
    assert step2.function.__name__ == "sum"
