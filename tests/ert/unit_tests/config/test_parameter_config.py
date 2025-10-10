import logging
import re
from pathlib import Path

import pytest
import xtgeo
from surfio import IrapSurface

from ert.config import (
    ExtParamConfig,
    Field,
    GenKwConfig,
    ParameterConfig,
    SurfaceConfig,
)
from ert.field_utils import Shape


def assert_parameters_in_logs(
    expected_provided_parameters: set,
    expected_defaulted_parameters: set,
    expected_parameter_class: ParameterConfig,
    caplog,
):
    def logged_properties_to_set(logged_properties: str):
        # The order of logged properties is arbitrary,
        # creating a set of the properties solves this
        return set(logged_properties.strip("'").split("', '"))

    parameter_log_pattern = (
        r"Attributes for (\S+) with input values:\n"
        r"\{([^}]+)\}\n"
        r"Attributes for (\S+) with defaulted values:\n"
        r"\{([^}]+)\}"
    )

    match = re.search(parameter_log_pattern, caplog.text)
    parameter_class = match.group(1)
    provided_parameters = match.group(2)
    unprovided_parameters = match.group(4)
    assert parameter_class == expected_parameter_class.__name__
    assert logged_properties_to_set(provided_parameters) == expected_provided_parameters
    assert (
        logged_properties_to_set(unprovided_parameters) == expected_defaulted_parameters
    )


@pytest.mark.parametrize("update", [True, False])
def test_parameters_are_logged_for_gen_kw_instances(caplog, update):
    config_list = [
        "KW_NAME",
        ("template.txt", "MY_KEYWORD <MY_KEYWORD>"),
        "kw.txt",
        ("prior.txt", "MY_KEYWORD LOGNORMAL 0 1"),
        {"UPDATE": "TRUE"} if update else {},
    ]
    caplog.set_level(logging.INFO)
    GenKwConfig.from_config_list(config_list)

    expected_attributes = {
        "name",
        "group",
        "distribution",
    }
    expected_defaults = {
        "type",
        "forward_init",
        "input_source",
    }
    if update:
        expected_attributes.add("update")
    else:
        expected_defaults.add("update")
    assert_parameters_in_logs(
        expected_attributes,
        expected_defaults,
        GenKwConfig,
        caplog,
    )


@pytest.mark.parametrize("update", [True, False])
def test_parameters_are_logged_for_surface_instances(caplog, monkeypatch, update):
    caplog.set_level(logging.INFO)

    class MockHeader:
        ncol = 0
        nrow = 0
        xori = 0
        yori = 0
        xinc = 0
        yinc = -1
        rot = 0

    class MockSurf:
        header = MockHeader()

    monkeypatch.setattr(Path, "exists", lambda _: True)
    monkeypatch.setattr(IrapSurface, "from_ascii_file", lambda _: MockSurf())
    options = {
        "INIT_FILES": "path/%dsurf.irap",
        "OUTPUT_FILE": "path/not_surface",
        "BASE_SURFACE": "surface/small_out.irap",
    }
    if update:
        options["UPDATE"] = "TRUE"
    SurfaceConfig.from_config_list(
        [
            "TOP",
            options,
        ]
    )
    expected_attributes = {
        "ncol",
        "rotation",
        "yflip",
        "nrow",
        "xinc",
        "yori",
        "forward_init_file",
        "base_surface_path",
        "xori",
        "yinc",
        "output_file",
        "name",
    }
    expected_defaults = {
        "type",
        "forward_init",
    }
    if update:
        expected_attributes.add("update")
    else:
        expected_defaults.add("update")
    assert_parameters_in_logs(
        expected_attributes,
        expected_defaults,
        SurfaceConfig,
        caplog,
    )


@pytest.mark.parametrize("update", [True, False])
def test_parameters_are_logged_for_field_instances(update, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    options = {
        "INIT_FILES": "foobar",
        "INIT_TRANSFORM": "LOG",
        "MIN": "10",
    }
    if update:
        options["UPDATE"] = "TRUE"
    grid_file_path = "Foo.egrid"

    def mock_grid(_):
        return xtgeo.create_box_grid(dimension=(Shape(3, 3, 3)))

    monkeypatch.setattr(xtgeo, "grid_from_file", mock_grid)

    config_list = ["Name", "", "output_filename.grdecl", options]
    Field.from_config_list(grid_file_path, config_list)
    expected_attributes = {
        "input_transformation",
        "truncation_min",
        "grid_file",
        "output_file",
        "file_format",
        "name",
        "forward_init_file",
        "ertbox_params",
    }
    defaulted_attributes = {
        "truncation_max",
        "type",
        "output_transformation",
        "forward_init",
        "mask_file",
    }
    if update:
        expected_attributes.add("update")
    else:
        defaulted_attributes.add("update")
    assert_parameters_in_logs(
        expected_attributes,
        defaulted_attributes,
        Field,
        caplog,
    )


def test_parameters_are_logged_for_ext_param_instances(caplog):
    caplog.set_level(logging.INFO)
    ExtParamConfig(
        name="Foo",
        output_file="Bar",
        update=False,
    )
    assert_parameters_in_logs(
        {"name", "output_file", "update"},
        {
            "forward_init",
            "forward_init_file",
            "input_keys",
            "type",
        },
        ExtParamConfig,
        caplog,
    )
