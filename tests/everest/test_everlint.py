import fileinput
import re
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError

from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from tests.everest.utils import everest_config_with_defaults


@pytest.mark.parametrize(
    "required_key",
    [
        "objective_functions",
        "controls",
        # "model", # This is not actually optional
        "config_path",
    ],
)
def test_missing_key(required_key, min_config):
    del min_config[required_key]
    errors = EverestConfig.lint_config_dict(min_config)
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"][0] == required_key


@pytest.mark.parametrize(
    "optional_key",
    [
        "output_constraints",
        "input_constraints",
        "install_jobs",
        "install_data",
        "forward_model",
        "simulator",
        "definitions",
    ],
)
def test_optional_keys(optional_key, min_config):
    assert optional_key not in min_config
    assert not EverestConfig.lint_config_dict(min_config)


def test_extra_key(min_config):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        everest_config_with_defaults(**min_config | {"extra": "extra"})


@pytest.mark.parametrize(
    "extra_config, expected",
    [
        ({"objective_functions": [{}]}, "Field required"),
        (
            {"input_constraints": [{"weights": {}}]},
            "(.*) weight data required for input constraints",
        ),
        (
            {"input_constraints": [{"weights": {"name": ["one, two"]}}]},
            "should be a valid number",
        ),
        (
            {"input_constraints": [{"weights": {"name": {"ans": 42}}}]},
            "should be a valid number",
        ),
        (
            {"input_constraints": [{"weights": {("one", "two"): 12}}]},
            "should be a valid string",
        ),
        (
            {"controls": [{"variables": []}]},
            "Value should have at least 1 item after validation, not 0",
        ),
        (
            {"config_path": "does_not_exist"},
            "no such file or directory .*/does_not_exist",
        ),
        (
            {
                "install_templates": [
                    {"template": "does_not_exist", "output_file": "not_relevant"}
                ]
            },
            "No such file or directory .*/does_not_exist",
        ),
        (
            {"model": {"realizations": [-1]}},
            "greater than or equal to 0",
        ),
        (
            {"model": {"realizations": ["apekatt"]}},
            "should be a valid integer",
        ),
        (
            {
                "install_data": [
                    {
                        "source": "not_relevant",
                        "target": ["Who am I?", "Not a string..."],
                    }
                ]
            },
            "target\n.* should be a valid string",
        ),
        (
            {"install_data": [{"source": None, "target": "not_relevant"}]},
            "Either source or data must be provided",
        ),
        (
            {
                "install_data": [
                    {"source": "", "data": {"foo": 1}, "target": "not_relevant"}
                ]
            },
            "The data and source options are mutually exclusive",
        ),
        (
            {"install_data": [{"source": ["a", "b"], "target": "not_relevant"}]},
            "source\n  Input should be a valid string",
        ),
        (
            {"install_data": [{"source": "not a file", "target": "not_relevant"}]},
            "No such file or directory",
        ),
        (
            {"install_jobs": [{"executable": 1, "name": "not_relevant"}]},
            "executable\n.* should be a valid string",
        ),
        (
            {"forward_model": ["not_a_job"]},
            "unknown job not_a_job",
        ),
        (
            {"environment": {"simulation_folder": "/usr/bin/unwriteable"}},
            "User does not have write access to",
        ),
        (
            {"environment": {"output_folder": ("super long path" * 300)}},
            "output_folder\n.* File name too long",
        ),
        (
            {"environment": {"output_folder": None}},
            "Value error, output_folder can not be None",
        ),
        (
            {"environment": {"output_folder": "/path/with/" + chr(0) + "embeddedNULL"}},
            "output_folder\n.* embedded null",
        ),
        (
            {"environment": {"output_folder": ["some", "list"]}},
            "output_folder\n.* str type expected",
        ),
        (
            {"input_constraints": [{"weights": {"not_exists": 1.0}}]},
            "not_exists.*not match any instance of control_name.variable_name",
        ),
    ],
)
def test_invalid_subconfig(extra_config, min_config, expected):
    for k, v in extra_config.items():
        min_config[k] = v
    with pytest.raises(ValidationError, match=expected):
        EverestConfig(**min_config)


@pytest.mark.parametrize(
    "link, source, target",
    [
        (True, "test_dir", "../test"),
        (False, "test_dir", "../test"),
        (True, "test_dir/my_file", "../test/test_file"),
        (False, "test_dir/my_file", "../test/test_file"),
    ],
)
def test_that_install_data_target_path_outside_runpath_is_invalid(
    link, source, target, tmp_path, monkeypatch, min_config
):
    monkeypatch.chdir(tmp_path)
    Path.mkdir(Path("test_dir"))
    Path("test_dir/my_file").touch()
    min_config["install_data"] = [{"source": source, "target": target, "link": link}]
    with pytest.raises(
        ValidationError,
        match=re.escape(f"Target location '{target}' is outside of the runpath."),
    ):
        EverestConfig(**min_config)


def test_no_list(min_config):
    min_config["install_data"] = []
    errors = EverestConfig.lint_config_dict(min_config)
    assert len(errors) == 0


def test_empty_list(min_config):
    min_config["install_data"] = []
    errors = EverestConfig.lint_config_dict(min_config)
    assert len(errors) == 0


@pytest.mark.parametrize(
    "value, valid",
    [
        (True, True),
        (False, True),
        (0, False),
        (1, False),
        ("True", False),
        (["I`m", []], False),
    ],
)
def test_bool_validation(value, valid, min_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    min_config["install_data"] = [
        {"source": "my_file", "target": "irrelephant", "link": value}
    ]
    expectation = (
        does_not_raise()
        if valid
        else pytest.raises(ValidationError, match="could not be parsed to a boolean")
    )
    with expectation:
        EverestConfig(**min_config)


def test_invalid_wells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match="cannot contain any dots"):
        everest_config_with_defaults(
            **yaml.safe_load(
                dedent("""
    model: {"realizations": [0] }
    wells: [{ name: fakename.fake}]
    definitions: {eclbase: my_test_case}
    """)
            )
        )


def test_well_ref_validation(min_config):
    config = min_config
    variables = config["controls"][0]["variables"]
    variables.append({"name": "a.new.well", "initial_guess": 0.2})
    with pytest.raises(ValueError, match="name can not contain any dots"):
        EverestConfig(**config)


def test_control_ref_validation(min_config):
    min_config["input_constraints"] = [{"weights": {"my_control.test": 1.0}}]
    EverestConfig(**min_config)


def test_that_control_group_name_contains_dot_is_linted(min_config):
    min_config["controls"][0]["name"] = "my.name"

    lint = EverestConfig.lint_config_dict(min_config)
    assert len(lint) == 1
    assert lint[0]["loc"] == ("controls", 0, "name")
    assert lint[0]["type"] == "value_error"


def test_that_missing_variables_field_in_control_group_is_linted(min_config):
    min_config["controls"][0].pop("variables")

    lint = EverestConfig.lint_config_dict(min_config)
    assert len(lint) == 1
    assert lint[0]["loc"] == ("controls", 0, "variables")
    assert lint[0]["type"] == "missing"


def test_that_invalid_type_for_control_group_item_is_linted(min_config):
    min_config["controls"][0] = "my vars"  # Modify the item at index 0

    lint = EverestConfig.lint_config_dict(min_config)
    assert len(lint) == 1
    assert lint[0]["loc"] == ("controls", 0)
    assert lint[0]["type"] == "model_type"


def test_that_missing_name_in_control_variable_is_linted(min_config):
    min_config["controls"][0]["variables"][0].pop("name")

    lints = EverestConfig.lint_config_dict(min_config)

    assert any(
        lint_
        for lint_ in lints
        if lint_["loc"]
        == ("controls", 0, "variables", "list[ControlVariableConfig]", 0, "name")
        and lint_["type"] == "missing"
    )


def test_that_invalid_type_for_control_variable_name_is_linted(min_config):
    min_config["controls"][0]["variables"][0]["name"] = {"name": True}

    lints = EverestConfig.lint_config_dict(min_config)
    assert any(
        lint_
        for lint_ in lints
        if lint_["loc"]
        == ("controls", 0, "variables", "list[ControlVariableConfig]", 0, "name")
        and lint_["type"] == "string_type"
    )


def test_that_control_variable_name_contains_dot_is_linted(min_config):
    min_config["controls"][0]["variables"][0]["name"] = "my.name"

    lints = EverestConfig.lint_config_dict(min_config)
    assert any(
        lint_
        for lint_ in lints
        if lint_["loc"]
        == ("controls", 0, "variables", "list[ControlVariableConfig]", 0, "name")
        and lint_["type"] == "value_error"
    )


@pytest.mark.parametrize(
    "target, expected",
    [
        (
            "r{{key1 }}/model/r{{key2}}.txt",
            r"The following keys are missing: \['r\{\{key1 \}\}', 'r\{\{key2\}\}'\]",
        ),
        (
            "r{{ key1}}/model/file.txt",
            r"The following key is missing: \['r\{\{ key1\}\}'\]",
        ),
        ("model/file.txt", ""),
    ],
)
def test_undefined_substitution(min_config, change_to_tmpdir, target, expected):
    config = min_config
    config["install_data"] = [
        {"source": "r{{configpath}}/../model/file.txt", "target": target}
    ]

    with open("config.yml", mode="w", encoding="utf-8") as f:
        yaml.dump(config, f)
    if expected:
        with pytest.raises(ValueError, match=expected) as e:
            yaml_file_to_substituted_config_dict("config.yml")

        print(e)
    else:
        yaml_file_to_substituted_config_dict("config.yml")


def test_commented_out_substitution(min_config, change_to_tmpdir):
    config = min_config
    config["forward_model"] = [
        "step1 abc",
        "step2 abc",
        "step3 abc",
    ]

    with open("config.yml", mode="w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with fileinput.input("config.yml", inplace=True) as fin:
        for line in fin:
            if "step2 abc" in line:
                print(" # - step2 r{{sub}} blabla")
            else:
                print(line, end="")

    config_dict = yaml_file_to_substituted_config_dict("config.yml")

    assert len(config_dict["forward_model"]) == 2
    assert config_dict["forward_model"][0] == "step1 abc"
    assert config_dict["forward_model"][1] == "step3 abc"


def test_eclbase_datafile_deprecation_message():
    lints = EverestConfig.lint_config_dict(
        {
            "definitions": {
                "eclbase": "/project/somewhere/everest/input/eclipse/include/EGGS"
            },
            "model": {"data_file": "the_data_file"},
        }
    )
    message = str(lints[0]["ctx"]["error"])

    assert (
        message
        == """
model.data_file is deprecated and will have no effect
to read summary data from forward model, do:
(replace flow with your chosen simulator forward model)
  forward_model:
    - job: flow
      results:
        file_name: /project/somewhere/everest/input/eclipse/include/EGGS
        type: summary
        keys: ['FOPR', 'WOPR']
""".strip()
    )


def test_that_geo_id_is_deprecated(min_config, change_to_tmpdir, caplog, capsys):
    config = min_config
    config["forward_model"] = ["step1 <GEO_ID>"]

    with open("config.yml", mode="w", encoding="utf-8") as f:
        yaml.dump(config, f)

    config_dict = yaml_file_to_substituted_config_dict("config.yml")

    assert config_dict["forward_model"][0] == "step1 <REALIZATION_ID>"
    assert "Deprecated key <GEO_ID> replaced by <REALIZATION_ID>." in caplog.text
    assert (
        r"<GEO_ID> is deprecated, please replace with 'r{{realization}}'."
        in capsys.readouterr().out
    )
