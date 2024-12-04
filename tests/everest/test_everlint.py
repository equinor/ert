import os
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError

from everest import ConfigKeys
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from tests.everest.test_config_validation import has_error
from tests.everest.utils import relpath

SNAKE_OIL_CONFIG = relpath("test_data/snake_oil/", "everest/model/snake_oil_all.yml")


@pytest.fixture
def min_config():
    yield yaml.safe_load(
        dedent("""
    model: {"realizations": [0]}
    controls:
      -
        name: my_control
        type: well_control
        min: 0
        max: 0.1
        variables:
          - { name: test, initial_guess: 0.1 }
    objective_functions:
      - {name: my_objective}
    config_path: .
    """)
    )


@pytest.mark.parametrize(
    "required_key",
    (
        ConfigKeys.OBJECTIVE_FUNCTIONS,
        ConfigKeys.CONTROLS,
        # ConfigKeys.MODEL, # This is not actually optional
        ConfigKeys.CONFIGPATH,
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
        ConfigKeys.OUTPUT_CONSTRAINTS,
        ConfigKeys.INPUT_CONSTRAINTS,
        ConfigKeys.INSTALL_JOBS,
        ConfigKeys.INSTALL_DATA,
        ConfigKeys.FORWARD_MODEL,
        ConfigKeys.SIMULATOR,
        ConfigKeys.DEFINITIONS,
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
    ],
)
def test_invalid_subconfig(extra_config, min_config, expected):
    for k, v in extra_config.items():
        min_config[k] = v
    with pytest.raises(ValidationError, match=expected):
        EverestConfig(**min_config)


def test_invalid_string():
    invalid_value = ["Who am I?", "Not a string.."]
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.INSTALL_DATA][0][ConfigKeys.TARGET] = invalid_value

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    has_error(errors, match="str type expected")


def test_no_list():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.INSTALL_DATA] = None

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0


def test_empty_list():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.INSTALL_DATA] = []

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0


def test_malformed_list():
    invalid_values = [["a", "b"], None, "not a file for sure?"]
    exp_errs = [
        "str type expected",
        "none is not an allowed value",
        "No such file or directory (.*)",
    ]

    for invalid_val, exp_err in zip(invalid_values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.INSTALL_DATA][0][ConfigKeys.SOURCE] = invalid_val

        errors = EverestConfig.lint_config_dict(config)

        assert len(errors) > 0
        has_error(errors, match=exp_err)


def test_no_installed_jobs():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config.pop(ConfigKeys.INSTALL_JOBS)

    errors = EverestConfig.lint_config_dict(config)
    for err in errors:
        has_error([err], match="unknown job (.*)")


def test_not_installed_job_in_script():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)

    config[ConfigKeys.FORWARD_MODEL] = ["my_well_drill"]

    errors = EverestConfig.lint_config_dict(config)
    has_error(errors, match="unknown job my_well_drill")


def test_validator_lint_value_error_msg():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.MODEL][ConfigKeys.REALIZATIONS][1] = -1  # invalid value
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 1
    has_error(errors, match="(.*) value is greater than or equal to 0")


def test_bool_validation():
    values = [True, False, 0, 1, "True", ["I'm", [True for real in []]]]
    exp_errs = 2 * [None] + 4 * ["(.*) could not be parsed to a boolean"]

    for val, exp_err in zip(values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.INSTALL_DATA][0][ConfigKeys.LINK] = val

        errors = EverestConfig.lint_config_dict(config)
        if exp_err is not None:
            has_error(errors, match=exp_err)


def test_simulation_spec():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.ENVIRONMENT][ConfigKeys.SIMULATION_FOLDER] = (
        "/usr/bin/unwriteable"
    )
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 1
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    has_error(errors, match="User does not have write access (.*)")

    config[ConfigKeys.ENVIRONMENT][ConfigKeys.SIMULATION_FOLDER] = (
        "/tmp/this_everest_folder/is/writeable"
    )
    errors = EverestConfig.lint_config_dict(config)
    assert not errors


def test_existing_path_validation():
    values = [
        SNAKE_OIL_CONFIG,
        os.path.split(SNAKE_OIL_CONFIG)[0],
        "A super path",
        0,
        None,
        ["I'm", "a", "path,", "not!"],
    ]

    exp_errs = (
        2 * [None]
        + 2 * ["No such file or directory (.*)"]
        + ["none is not an allowed value"]
        + ["str type expected"]
    )

    for val, exp_err in zip(values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.INSTALL_DATA][0][ConfigKeys.SOURCE] = val

        errors = EverestConfig.lint_config_dict(config)
        if exp_err is None:
            assert len(errors) == 0
        else:
            has_error(errors, match=exp_err)


def test_existing_file_validation():
    well_job = os.path.join(
        os.path.dirname(SNAKE_OIL_CONFIG), "..", "..", "jobs/SNAKE_OIL_NPV"
    )
    values = [
        well_job,
        os.path.split(SNAKE_OIL_CONFIG)[0],
        "A super path",
        0,
        None,
        ["I'm", "a", "path,", "not!"],
    ]
    exp_errs = (
        [None]
        + ["Is not a file (.*)"]
        + 2 * ["No such file or directory (.*)"]
        + ["none is not an allowed value"]
        + ["str type expected"]
    )

    for val, exp_err in zip(values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        jobs = config[ConfigKeys.INSTALL_JOBS]
        jobs[0][ConfigKeys.SOURCE] = val

        errors = EverestConfig.lint_config_dict(config)
        if exp_err is None:
            assert len(errors) == 0
        else:
            has_error(errors, match=exp_err)


def test_valid_path_validation():
    values = [
        SNAKE_OIL_CONFIG,
        os.path.dirname(SNAKE_OIL_CONFIG),
        "A super path",
        ("super long path" * 300),
        0,
        None,
        ["I'm", "a", "path,", "not!"],
        "/path/with/" + chr(0) + "embeddedNULL",
    ]

    exp_errs = (
        [None, None, None]
        + ["File name too long"]
        + 3 * ["str type expected"]
        + ["embedded null byte"]
    )

    for val, exp_err in zip(values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.ENVIRONMENT][ConfigKeys.OUTPUT_DIR] = val

        errors = EverestConfig.lint_config_dict(config)
        if exp_err is None:
            assert len(errors) == 0
        else:
            has_error(errors, match=exp_err)


def test_valid_filepath_validation():
    values = [
        os.path.dirname(SNAKE_OIL_CONFIG),
        os.path.dirname(SNAKE_OIL_CONFIG) + "/export.csv",
        "path/to/file.csv",
        "path/to/folder/",
    ]

    exp_errs = ["Invalid type", None, None, "Invalid type"]

    for val, exp_err in zip(values, exp_errs, strict=False):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config["export"] = {}
        config["export"]["csv_output_filepath"] = val
        errors = EverestConfig.lint_config_dict(config)

        if exp_err is None:
            assert len(errors) == 0
        else:
            has_error(errors, match=exp_err)


def test_ert_job_file():
    content = [
        ("ARGUMENT 1\n", "missing EXECUTABLE (.*)"),
        ("EXECUTABLE /no/such/path\n", "No such executable (.*)"),
        ("EXECUTABLE %s\n" % SNAKE_OIL_CONFIG, "(.*)_oil_all.yml is not executable"),
    ]
    for cnt, err in content:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as f:
            f.write(cnt)
            config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
            config[ConfigKeys.INSTALL_JOBS][0][ConfigKeys.SOURCE] = f.name

            errors = EverestConfig.lint_config_dict(config)
            has_error(errors, match=err)


def test_well_ref_validation():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0

    variables = config[ConfigKeys.CONTROLS][0][ConfigKeys.VARIABLES]
    variables.append({ConfigKeys.NAME: "a.new.well", ConfigKeys.INITIAL_GUESS: 0.2})
    errors = EverestConfig.lint_config_dict(config)
    has_error(errors, match="(.*) name can not contain any dots")

    wells = config[ConfigKeys.WELLS]
    wells.append({ConfigKeys.NAME: "a.new.well"})
    errors = EverestConfig.lint_config_dict(config)
    has_error(errors, match="(.*) name can not contain any dots")

    variables[-1][ConfigKeys.NAME] = "aNewWell"
    wells[-1] = {ConfigKeys.NAME: "aNewWell"}
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0


def test_control_ref_validation():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0

    weights = config[ConfigKeys.INPUT_CONSTRAINTS][0][ConfigKeys.WEIGHTS]
    weights.pop("group.W1")

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 0

    weights["no_group.W1"] = 1.0
    weights["group.no_ctrl"] = 1.0
    weights["g.n.c.s.w"] = 1.0
    weights["asdasdasd"] = 1.0

    errors = EverestConfig.lint_config_dict(config)
    errors = errors[0]["ctx"]["error"].args[0]
    assert len(errors) == 4
    for err in errors:
        assert "not match any instance of control_name.variable_name" in err


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
        config.pop(ConfigKeys.CONTROLS)
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS] = "monkey"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up control group name
        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS][0].pop(ConfigKeys.NAME)
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS][0]["name"] = ["my", "name"]
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS][0]["name"] = "my.name"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up variables
        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS][0].pop(ConfigKeys.VARIABLES)
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        config[ConfigKeys.CONTROLS][0] = "my vars"
        assert len(EverestConfig.lint_config_dict(config)) > 0

        # Messed up names
        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config[ConfigKeys.CONTROLS][0][ConfigKeys.VARIABLES][0]
        variable.pop(ConfigKeys.NAME)
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config[ConfigKeys.CONTROLS][0][ConfigKeys.VARIABLES][0]
        variable[ConfigKeys.NAME] = {"name": True}
        assert len(EverestConfig.lint_config_dict(config)) > 0

        config = yaml_file_to_substituted_config_dict(config_file)
        variable = config[ConfigKeys.CONTROLS][0][ConfigKeys.VARIABLES][0]
        variable[ConfigKeys.NAME] = "my.name"
        assert len(EverestConfig.lint_config_dict(config)) > 0


def test_date_type():
    valid_dates = (
        "2000-1-1",
        "2010-1-1",
        "2018-12-31",
    )

    invalid_dates = (
        "32.01.2000",
        "2000-1-32",
        "fdsafdas",
        "01-01-01",
        "...",
        None,
        {},
        "2000-2-30",
    )

    for date in valid_dates + invalid_dates:
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.WELLS][0][ConfigKeys.DRILL_DATE] = date

        err = EverestConfig.lint_config_dict(config)
        if date in valid_dates:
            assert not err
        else:
            assert len(err) > 0, "%s was wrongly accepted" % date
            has_error(err, match=f"malformed date: {date}(.*)")


@pytest.mark.fails_on_macos_github_workflow
def test_lint_everest_models_jobs():
    pytest.importorskip("everest_models")
    config_file = relpath("../../test-data/everest/egg/everest/model/config.yml")
    config = EverestConfig.load_file(config_file).to_dict()
    # Check initial config file is valid
    assert len(EverestConfig.lint_config_dict(config)) == 0
