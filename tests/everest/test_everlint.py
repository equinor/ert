import os
import tempfile
from pathlib import Path

import pytest

import everest
from everest import ConfigKeys
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.util.forward_models import collect_forward_models
from tests.everest.test_config_validation import has_error
from tests.everest.utils import relpath

REQUIRED_TOP_KEYS = (
    ConfigKeys.OBJECTIVE_FUNCTIONS,
    ConfigKeys.CONTROLS,
)
OPTIONAL_TOP_KEYS = (
    ConfigKeys.OUTPUT_CONSTRAINTS,
    ConfigKeys.INPUT_CONSTRAINTS,
    ConfigKeys.INSTALL_JOBS,
    ConfigKeys.INSTALL_DATA,
    ConfigKeys.FORWARD_MODEL,
    ConfigKeys.SIMULATOR,
    ConfigKeys.DEFINITIONS,
)


def _key(lint_msg):
    return lint_msg.key


def _keys(lint_msgs):
    return list(map(_key, lint_msgs))


SNAKE_OIL_CONFIG = relpath("test_data/snake_oil/", "everest/model/snake_oil_all.yml")


def test_missing_key():
    for key in REQUIRED_TOP_KEYS:
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        del config[key]

        if key == ConfigKeys.CONTROLS:
            del config[ConfigKeys.INPUT_CONSTRAINTS]  # avoid ctrl ref err
        errors = EverestConfig.lint_config_dict(config)
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"


def test_removing_optional_key():
    for key in OPTIONAL_TOP_KEYS:
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)

        if key in config:
            config.pop(key)

            # NOTE: Installing jobs are optional, but not if you want to run
            # imported jobs! At least until Everest comes with a default set of
            # jobs.
            if key == ConfigKeys.INSTALL_JOBS:
                config.pop(ConfigKeys.FORWARD_MODEL)

            assert len(EverestConfig.lint_config_dict(config)) == 0


def test_extra_key():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config["EXTRA_KEY"] = "funcy data for the win"

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 1
    has_error(errors, match="extra fields not permitted")


def test_no_data():
    # empty required dict
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.OBJECTIVE_FUNCTIONS].append({})
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    assert has_error(errors, match="Field required")  # no name

    # empty required shallow dict
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.INPUT_CONSTRAINTS][0][ConfigKeys.WEIGHTS] = {}
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    assert has_error(errors, match="(.*) weight data required for input constraints")

    # empty required list
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.CONTROLS][0][ConfigKeys.VARIABLES] = []
    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    assert has_error(
        errors, match="Value should have at least 1 item after validation, not 0"
    )


def test_invalid_shallow_value():
    for invalid_val in [["one", "two"], {"ans": 42}]:
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        input_constr_weights = config[ConfigKeys.INPUT_CONSTRAINTS][0]
        input_constr_weights = input_constr_weights[ConfigKeys.WEIGHTS]
        input_constr_weights[next(iter(input_constr_weights.keys()))] = invalid_val

        errors = EverestConfig.lint_config_dict(config)
        has_error(errors, match="value is not a valid float")
        assert len(errors) > 0
        assert len(errors) == 1


def test_invalid_shallow_key():
    invalid_key = ("one", "two")
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    input_constr_weights = config[ConfigKeys.INPUT_CONSTRAINTS][0]
    input_constr_weights = input_constr_weights[ConfigKeys.WEIGHTS]
    input_constr_weights[invalid_key] = 12

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) == 1
    has_error(errors, match="str type expected")


def test_non_existent_dir():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)

    config[ConfigKeys.CONFIGPATH] = Path(
        config[ConfigKeys.CONFIGPATH] + "fndjffdsn/is/no/dir/"
    )

    retrieved_dir = str(config[ConfigKeys.CONFIGPATH].parent)
    assert bool(retrieved_dir)
    assert not os.path.isdir(retrieved_dir)

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    has_error(errors, match="No such file or directory (.*)")


def test_non_existent_file():
    non_existing_file = "fndjffdsn/is/no/file"
    assert not os.path.isfile(non_existing_file)

    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    config[ConfigKeys.INSTALL_TEMPLATES] = [
        {
            ConfigKeys.TEMPLATE: non_existing_file,
            ConfigKeys.OUTPUT_FILE: "fndjffdsn/whatever/file",
        }
    ]

    errors = EverestConfig.lint_config_dict(config)
    assert len(errors) > 0
    has_error(errors, match="No such file or directory (.*)")


def test_invalid_integer():
    invalid_values = [-1, -999, "apekatt"]
    exp_errors = 2 * ["(.*)greater than or equal to 0"] + ["(.*) not a valid integer"]
    for invalid_value, err in zip(invalid_values, exp_errors):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.MODEL][ConfigKeys.REALIZATIONS][1] = invalid_value

        errors = EverestConfig.lint_config_dict(config)
        assert len(errors) > 0
        has_error(errors, match=err)


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

    for invalid_val, exp_err in zip(invalid_values, exp_errs):
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

    for val, exp_err in zip(values, exp_errs):
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

    for val, exp_err in zip(values, exp_errs):
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

    for val, exp_err in zip(values, exp_errs):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        jobs = config[ConfigKeys.INSTALL_JOBS]
        jobs[0][ConfigKeys.SOURCE] = val

        errors = EverestConfig.lint_config_dict(config)
        if exp_err is None:
            assert len(errors) == 0
        else:
            has_error(errors, match=exp_err)


def test_existing_dir_validation():
    values = [
        SNAKE_OIL_CONFIG,
        "I'm not a path!",
    ]

    exp_errs = [None, "no such file or directory (.*)"]

    for val, exp_err in zip(values, exp_errs):
        config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
        config[ConfigKeys.CONFIGPATH] = Path(val)
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

    for val, exp_err in zip(values, exp_errs):
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

    for val, exp_err in zip(values, exp_errs):
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


def test_default_jobs():
    config_file = relpath("test_data/mocked_test_case/mocked_test_case.yml")
    config = EverestConfig.load_file(config_file)
    config.forward_model += everest.jobs.script_names
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0


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


def test_lint_report_steps():
    config_file = relpath("test_data/mocked_test_case/mocked_test_case.yml")
    config = EverestConfig.load_file(config_file).to_dict()
    # Check initial config file is valid
    assert len(EverestConfig.lint_config_dict(config)) == 0
    config[ConfigKeys.MODEL][ConfigKeys.REPORT_STEPS] = [
        "2000-1-1",
        "2001-1-1",
        "2002-1-1",
        "2003-1-1",
    ]
    # Check config file is valid after report steps have been added
    assert len(EverestConfig.lint_config_dict(config)) == 0
    config[ConfigKeys.MODEL][ConfigKeys.REPORT_STEPS].append("invalid_date")
    # Check config no longer valid when invalid date is added
    errors = EverestConfig.lint_config_dict(config)
    has_error(errors, match="malformed date: invalid_date(.*)")


@pytest.mark.fails_on_macos_github_workflow
def test_lint_everest_models_jobs():
    pytest.importorskip("everest_models")
    config_file = relpath("../../test-data/everest/egg/everest/model/config.yml")
    config = EverestConfig.load_file(config_file).to_dict()
    # Check initial config file is valid
    assert len(EverestConfig.lint_config_dict(config)) == 0


def test_overloading_everest_models_names():
    config = yaml_file_to_substituted_config_dict(SNAKE_OIL_CONFIG)
    for job in collect_forward_models():
        config["install_jobs"][2]["name"] = job["name"]
        config["forward_model"][1] = job["name"]
        errors = EverestConfig.lint_config_dict(config)
        assert len(errors) == 0, f"Failed for job {job['name']}"
