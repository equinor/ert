import pytest

from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from tests.everest.test_config_validation import has_error
from tests.everest.utils import relpath


@pytest.fixture
def mocked_config():
    return yaml_file_to_substituted_config_dict(
        relpath("test_data", "mocked_test_case", "mocked_test_case.yml")
    )


def test_lint_mocked_config(mocked_config):
    EverestConfig(**mocked_config)


def test_unrecognized_keys(mocked_config):
    config = mocked_config
    wells = config["wells"]

    wells[0]["unexpected_key"] = 7
    assert has_error(
        EverestConfig.lint_config_dict(config), match="Extra inputs are not permitted"
    )
    wells[0].pop("unexpected_key")
    wells[0][14] = 7
    assert has_error(
        EverestConfig.lint_config_dict(config), match=".*Keys should be strings"
    )

    wells[0].pop(14)
    assert len(EverestConfig.lint_config_dict(config)) == 0


def test_well_names(mocked_config):
    config = mocked_config
    wells = config["wells"]
    assert len(wells) == 16

    well_name_0 = wells[0]["name"]
    well_name_1 = wells[1]["name"]

    # Make sure controls do not interfere with error checking
    controls = config["controls"][0]["variables"]
    for i, ctrl in enumerate(controls):
        if ctrl["name"] == well_name_0:
            del controls[i]
            break

    wells[0].pop("name")  # empty well definition
    assert has_error(EverestConfig.lint_config_dict(config), match="Field required")

    wells[0]["drill_time"] = 1  # incomplete well definition
    assert has_error(EverestConfig.lint_config_dict(config), match="Field required")

    wells[0]["name"] = well_name_1  # not unique name
    assert has_error(
        EverestConfig.lint_config_dict(config), match="Well names must be unique"
    )

    wells[0]["name"] = "a.b"  # can't have dots in well name
    assert has_error(
        EverestConfig.lint_config_dict(config),
        match="Well name can not contain any dots (.)",
    )

    wells[0]["name"] = well_name_0
    assert not EverestConfig.lint_config_dict(config)


def test_well_drilling_times(mocked_config):
    config = mocked_config
    wells = config["wells"]

    wells[0]["drill_time"] = -4
    assert has_error(
        EverestConfig.lint_config_dict(config),
        match="Drill time must be a positive number",
    )

    wells[0]["drill_time"] = 0
    assert has_error(
        EverestConfig.lint_config_dict(config),
        match="Drill time must be a positive number",
    )
    wells[0]["drill_time"] = "seventeen"
    assert has_error(
        EverestConfig.lint_config_dict(config),
        match="Input should be a valid number",
    )

    wells[0]["drill_time"] = 3.7
    assert not EverestConfig.lint_config_dict(config)
