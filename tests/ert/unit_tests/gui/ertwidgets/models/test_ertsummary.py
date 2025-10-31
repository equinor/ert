from unittest.mock import MagicMock

import pytest

from ert.config import ErtConfig, Field, GenKwConfig, SurfaceConfig
from ert.field_utils import FieldFileFormat, GridGeometry
from ert.gui.ertwidgets import ErtSummary


@pytest.fixture
def mock_ert():
    ert_mock = MagicMock()

    ert_mock.forward_model_step_name_list.return_value = [
        "forward_model_1",
        "forward_model_2",
    ]

    gen_kws = {
        "KEY_1": GenKwConfig(
            name="KEY_1",
            distribution={"name": "uniform", "min": 0, "max": 1},
        ),
        "KEY_2": GenKwConfig(
            name="KEY_2",
            distribution={"name": "normal", "mean": 0, "std": 1},
        ),
        "KEY_3": GenKwConfig(
            name="KEY_3",
            distribution={"name": "lognormal", "mean": 0, "std": 1},
        ),
    }

    surface = SurfaceConfig(
        name="surface",
        forward_init=True,
        ncol=10,
        nrow=7,
        xori=1,
        yori=1,
        xinc=1,
        yinc=1,
        rotation=1,
        yflip=1,
        forward_init_file="input_%d",
        output_file="output",
        base_surface_path="base_surface",
        update=True,
    )

    grid_geometry = GridGeometry(
        nx=10,
        ny=5,
        nz=3,
        xlength=1,
        ylength=1,
        xinc=1,
        yinc=1,
        rotation_angle=45,
        origin=(0, 0),
    )

    field = Field(
        name="field",
        forward_init=True,
        grid_geometry=grid_geometry,
        file_format=FieldFileFormat.ROFF,
        output_transformation=None,
        input_transformation=None,
        truncation_min=None,
        truncation_max=None,
        forward_init_file="",
        output_file="",
        grid_file="",
        update=True,
    )

    ert_mock.ensemble_config.parameter_configs = {
        "surface": surface,
        "field": field,
    } | gen_kws

    ert_mock.parameter_configurations_with_design_matrix = list(
        ert_mock.ensemble_config.parameter_configs.values()
    )

    return ert_mock


def test_getForwardModels(mock_ert):
    expected_list = ["forward_model_1", "forward_model_2"]
    forward_model_list = ErtSummary(mock_ert).getForwardModels()
    assert forward_model_list == expected_list


def test_getParameters(mock_ert):
    expected_list = ["field (10, 5, 3)", "gen_kw (3)", "surface (10, 7)"]
    parameter_list, parameter_count = ErtSummary(mock_ert).get_parameters()
    assert parameter_list == expected_list
    assert parameter_count == 223


def test_that_design_matrix_parameters_are_included_in_the_parameter_count(mock_ert):
    # Add design matrix parameters
    dm_param1 = GenKwConfig(
        name="dm_param_a",
        distribution={"name": "uniform", "min": 0, "max": 1},
        group="DESIGN_MATRIX",
    )
    dm_param2 = GenKwConfig(
        name="dm_param_b",
        distribution={"name": "uniform", "min": 0, "max": 1},
        group="DESIGN_MATRIX",
    )
    dm_param3 = GenKwConfig(
        name="dm_param_c",
        distribution={"name": "uniform", "min": 0, "max": 1},
        group="DESIGN_MATRIX",
    )

    # Modify the mock to return parameters including design matrix
    mock_ert.parameter_configurations_with_design_matrix = [
        *mock_ert.ensemble_config.parameter_configs.values(),
        dm_param1,
        dm_param2,
        dm_param3,
    ]

    parameter_list, parameter_count = ErtSummary(mock_ert).get_parameters()

    # Check that design matrix parameters are counted
    assert "DESIGN_MATRIX (3)" in parameter_list
    # Original count (223) + 3 design matrix parameters
    assert parameter_count == 226


@pytest.mark.usefixtures("use_tmpdir")
def test_design_matrix_parameters_counted_when_loaded_from_real_config(
    copy_poly_case_with_design_matrix,
):
    # Create a design matrix with 3 parameters (a, b, c)
    design_dict = {"REAL": [0, 1, 2], "a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    copy_poly_case_with_design_matrix(design_dict, [])

    config = ErtConfig.from_file("poly.ert")
    summary = ErtSummary(config)

    parameter_list, parameter_count = summary.get_parameters()

    # Should have 3 parameters from design matrix (a, b, c)
    assert "DESIGN_MATRIX (3)" in parameter_list
    assert parameter_count == 3


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_snake_oil(snake_oil_case):
    summary = ErtSummary(snake_oil_case)

    assert summary.getForwardModels() == [
        "SNAKE_OIL_SIMULATOR",
        "SNAKE_OIL_NPV",
        "SNAKE_OIL_DIFF",
    ]

    assert summary.get_parameters() == (["SNAKE_OIL_PARAM (10)"], 10)

    assert summary.getObservations() == [
        {"observation_key": "WOPR_OP1_108", "count": 1},
        {"observation_key": "WOPR_OP1_144", "count": 1},
        {"observation_key": "WOPR_OP1_190", "count": 1},
        {"observation_key": "WOPR_OP1_36", "count": 1},
        {"observation_key": "WOPR_OP1_72", "count": 1},
        {"observation_key": "WOPR_OP1_9", "count": 1},
        {"observation_key": "WPR_DIFF_1", "count": 4},
    ]
