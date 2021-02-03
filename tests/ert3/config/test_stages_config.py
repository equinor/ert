import os
import pathlib
import pydantic
import pytest
import shutil

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
        [[{"not_a_key": "value"}], "3 validation errors"],
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
                "input": [{"record": "some_record", "location": "some_location"}],
                "output": [{"record": "some_record", "location": "some_location"}],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": ["poly --help"],
            }
        ]
    )
    assert config[0].name == "some_name"
    assert config[0].script[0] == "poly --help"


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
    config = ert3.config.load_stages_config(config)


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


def test_stages_get_script():
    config = ert3.config.load_stages_config(
        [
            {
                "name": "some_name",
                "input": [{"record": "some_record", "location": "some_file"}],
                "output": [{"record": "some_record", "location": "some_file"}],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": [
                    "poly --coefficients coefficients.json --output output.json"
                ],
            }
        ]
    )
    step = config.step_from_key("some_name")
    assert step.script == ["poly --coefficients coefficients.json --output output.json"]
