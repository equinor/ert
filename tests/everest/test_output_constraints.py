import os

import pytest
from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType

from everest import ConfigKeys
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.suite import _EverestWorkflow

from .test_config_validation import has_error

CONFIG_FILE = "config_output_constraints.yml"


def test_constraints_init(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    constr = list(config.output_constraints or [])

    constr_names = [cn.name for cn in constr]
    assert constr_names == [
        "oil_prod_rate_000",
        "oil_prod_rate_001",
        "oil_prod_rate_002",
        "oil_prod_rate_003",
        "oil_prod_rate_004",
        "oil_prod_rate_005",
        "oil_prod_rate_006",
        "oil_prod_rate_007",
        "oil_prod_rate_008",
        "oil_prod_rate_009",
        "oil_prod_rate_010",
        "oil_prod_rate_011",
        "oil_prod_rate_012",
        "oil_prod_rate_013",
        "oil_prod_rate_014",
        "oil_prod_rate_015",
    ]

    assert [cn.upper_bound for cn in constr] == 16 * [5000]
    assert [cn.scale for cn in constr] == 16 * [7500]


def test_wrong_output_constr_def(copy_mocked_test_data_to_tmp):
    # No RHS
    errors = EverestConfig.lint_config_dict(
        {
            "wells": [{"name": "w07"}],
            "output_constraints": [
                {"name": "some_name"},
            ],
            "config_path": "/",
            "forward_model": [],
            "controls": [
                {
                    "name": "well_order",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [{"name": "w07", "initial_guess": 0.0633}],
                }
            ],
            "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
            "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
            "model": {ConfigKeys.REALIZATIONS: [0]},
            "objective_functions": [{"name": "npv_function"}],
        }
    )

    assert has_error(
        errors, match="Output constraints must have only one of the following"
    )

    # Same name
    errors = EverestConfig.lint_config_dict(
        {
            "wells": [{"name": "w07"}],
            "output_constraints": [
                {"name": "same_name", "upper_bound": 5000},
                {"name": "same_name", "upper_bound": 5000},
            ],
            "config_path": "/",
            "forward_model": [],
            "controls": [
                {
                    "name": "well_order",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [{"name": "w07", "initial_guess": 0.0633}],
                }
            ],
            "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
            "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
            "model": {ConfigKeys.REALIZATIONS: [0]},
            "objective_functions": [{"name": "npv_function"}],
        }
    )
    assert has_error(errors, match="Output constraint names must be unique")

    # Two RHS
    errors = EverestConfig.lint_config_dict(
        {
            "wells": [{"name": "w07"}],
            "output_constraints": [
                {"name": "some_name", "upper_bound": 5000, "target": 5000},
            ],
            "config_path": "/",
            "forward_model": [],
            "controls": [
                {
                    "name": "well_order",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [{"name": "w07", "initial_guess": 0.0633}],
                }
            ],
            "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
            "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
            "model": {ConfigKeys.REALIZATIONS: [0]},
            "objective_functions": [{"name": "npv_function"}],
        }
    )
    assert has_error(
        errors,
        match="Output constraints must have only one of the following:"
        " { target }, or { upper and/or lower bound }",
    )

    # Wrong RHS attribute
    wrong_rhs_config = {
        "wells": [{"name": "w07"}],
        "output_constraints": [
            {"name": "some_name", "target": 2},
        ],
        "config_path": "/",
        "forward_model": [],
        "controls": [
            {
                "name": "well_order",
                "type": "well_control",
                "min": 0,
                "max": 1,
                "variables": [{"name": "w07", "initial_guess": 0.0633}],
            }
        ],
        "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
        "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
        "model": {ConfigKeys.REALIZATIONS: [0]},
        "objective_functions": [{"name": "npv_function"}],
    }

    wrong_rhs_config["output_constraints"][0]["upper_bund"] = 5000

    errors = EverestConfig.lint_config_dict(wrong_rhs_config)
    assert has_error(errors, match="Extra inputs are not permitted")

    # Wrong RHS type
    errors = EverestConfig.lint_config_dict(
        {
            "wells": [{"name": "w07"}],
            "output_constraints": [
                {"name": "some_name", "upper_bound": "2ooo"},
            ],
            "config_path": "/",
            "forward_model": [],
            "controls": [
                {
                    "name": "well_order",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [{"name": "w07", "initial_guess": 0.0633}],
                }
            ],
            "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
            "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
            "model": {ConfigKeys.REALIZATIONS: [0]},
            "objective_functions": [{"name": "npv_function"}],
        }
    )
    assert has_error(errors, "unable to parse string as a number")


def test_upper_bound_output_constraint_def(copy_mocked_test_data_to_tmp):
    with open("conf_file", "w", encoding="utf-8") as f:
        f.write(" ")

    config = EverestConfig.with_defaults(
        **{
            "wells": [{"name": "w07"}],
            "output_constraints": [
                {"name": "some_name", "upper_bound": 5000, "scale": 1.0},
            ],
            "config_path": os.getcwd() + "/conf_file",
            "forward_model": [],
            "controls": [
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [{"name": "w07", "initial_guess": 0.0633}],
                }
            ],
            "environment": {ConfigKeys.SIMULATION_FOLDER: "/tmp/everest"},
            "optimization": {ConfigKeys.ALGORITHM: "optpp_q_newton"},
            "model": {ConfigKeys.REALIZATIONS: [0]},
            "objective_functions": [{"name": "npv_function"}],
        }
    )

    # Check ropt conversion
    ropt_conf = EnOptConfig.model_validate(everest2ropt(config))

    expected = {
        "scale": 1.0,
        "name": "some_name",
        "rhs_value": [5000],
        "type": ConstraintType.LE,
    }

    assert expected["scale"] == 1.0 / ropt_conf.nonlinear_constraints.scales[0]
    assert expected["name"] == ropt_conf.nonlinear_constraints.names[0]
    assert expected["rhs_value"] == ropt_conf.nonlinear_constraints.rhs_values[0]
    assert expected["type"] == ropt_conf.nonlinear_constraints.types[0]

    workflow = _EverestWorkflow(config)
    assert workflow is not None


@pytest.mark.integration_test
def test_sim_output_constraints(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()
