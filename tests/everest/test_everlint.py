from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError

from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from tests.everest.test_config_validation import has_error
from tests.everest.utils import relpath


@pytest.mark.parametrize(
    "required_key",
    (
        "objective_functions",
        "controls",
        # "model", # This is not actually optional
        "config_path",
    ),
)
def test_missing_key(required_key, min_config):
    del min_config[required_key]
    errors = EverestConfig.lint_config_dict(min_config)
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"][0] == required_key


@pytest.mark.parametrize(
    "optional_key",
    (
        "output_constraints",
        "input_constraints",
        "install_jobs",
        "install_data",
        "forward_model",
        "simulator",
        "definitions",
    ),
)
def test_optional_keys(optional_key, min_config):
    assert optional_key not in min_config
    assert not EverestConfig.lint_config_dict(min_config)


def test_extra_key(min_config):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EverestConfig.with_defaults(**min_config | {"extra": "extra"})


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
            "source\n  Input should be a valid string",
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
            {"install_jobs": [{"source": None, "name": "not_relevant"}]},
            "source\n.* should be a valid string",
        ),
        (
            {"forward_model": ["not_a_job"]},
            "unknown job not_a_job",
        ),
        (
            {"model": {"realizations": [-1]}},
            "greater than or equal to 0",
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
            "output_folder\n.* str type expected",
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


def test_no_list(min_config):
    min_config["install_data"] = None
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


@pytest.mark.parametrize(
    "path_val, valid", [("my_file", True), ("my_folder/", False), ("my_folder", False)]
)
def test_export_filepath_validation(min_config, tmp_path, monkeypatch, path_val, valid):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    Path("my_folder").mkdir()
    min_config["export"] = {"csv_output_filepath": path_val}
    expectation = (
        does_not_raise()
        if valid
        else pytest.raises(ValidationError, match="Invalid type")
    )
    with expectation:
        EverestConfig(**min_config)


def test_invalid_wells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    with pytest.raises(ValidationError, match="can not contain any dots"):
        EverestConfig.with_defaults(
            **yaml.safe_load(
                dedent("""
    model: {"realizations": [0], data_file: my_file}
    wells: [{ name: fakename.fake}]
    definitions: {eclbase: my_test_case}
    """)
            )
        )


def test_well_ref_validation(min_config):
    config = min_config
    variables = config["controls"][0]["variables"]
    variables.append({"name": "a.new.well", "initial_guess": 0.2})
    errors = EverestConfig.lint_config_dict(config)
    has_error(errors, match="(.*) name can not contain any dots")


def test_control_ref_validation(min_config):
    min_config["input_constraints"] = [{"weights": {"my_control.test": 1.0}}]
    EverestConfig(**min_config)


@pytest.mark.integration_test
def test_init_context_controls():
    test_configs = [
        "test_data/mocked_test_case/config_input_constraints.yml",
        "test_data/mocked_test_case/mocked_test_case.yml",
    ]
    test_configs = map(relpath, test_configs)

    for config_file in test_configs:
        # No initial errors
        config = yaml_file_to_substituted_config_dict(config_file)
        assert len(EverestConfig.lint_config_dict(config)) == 0

        # Messed up controls
        config = yaml_file_to_substituted_config_dict(config_file)
        config.pop("controls")
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"] = "monkey"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up control group name
        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"][0].pop("name")
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"][0]["name"] = ["my", "name"]
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"][0]["name"] = "my.name"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up variables
        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"][0].pop("variables")
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config["controls"][0] = "my vars"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up names
        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config["controls"][0]["variables"][0]
        variable.pop("name")
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config["controls"][0]["variables"][0]
        variable["name"] = {"name": True}
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config["controls"][0]["variables"][0]
        variable["name"] = "my.name"
        assert len(EverestConfig.lint_config_dict(config)) > 0


@pytest.mark.parametrize(
    "date, valid",
    [
        ("2000-1-1", True),
        ("2010-1-1", True),
        ("2018-12-31", True),
        ("32.01.2000", False),
        ("2000-1-32", False),
        ("fdsafdas", False),
        ("01-01-01", False),
        ("...", False),
        (None, False),
        ("2000-2-30", False),
    ],
)
def test_date_type(date, valid, min_config):
    expectation = (
        does_not_raise()
        if valid
        else pytest.raises(ValidationError, match=f"malformed date: {date}")
    )
    min_config["wells"] = [{"drill_date": date, "name": "test"}]
    with expectation:
        EverestConfig(**min_config)


@pytest.mark.fails_on_macos_github_workflow
def test_lint_everest_models_jobs():
    pytest.importorskip("everest_models")
    config_file = relpath("../../test-data/everest/egg/everest/model/config.yml")
    config = EverestConfig.load_file(config_file).to_dict()
    # Check initial config file is valid
    assert len(EverestConfig.lint_config_dict(config)) == 0
