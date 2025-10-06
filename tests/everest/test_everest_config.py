import logging
from pathlib import Path
from textwrap import dedent

import pytest

from everest.config import (
    EverestConfig,
    EverestValidationError,
    ServerConfig,
    SimulatorConfig,
)
from everest.config.control_config import ControlConfig
from everest.config.control_variable_config import ControlVariableConfig
from everest.config.cvar_config import CVaRConfig
from everest.config.optimization_config import OptimizationConfig


def test_that_str_type_failures_are_propagated(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    Path("everest_config.yml").write_text(
        dedent("""
        objective_functions:
            - name:
                job: distance
    """),
        encoding="utf-8",
    )

    with pytest.raises(EverestValidationError) as err:
        EverestConfig.load_file("everest_config.yml")

    assert any(
        e for e in err.value.errors if "Input should be a valid string" in e["msg"]
    )


def test_that_control_config_is_initialized_with_control_variables():
    controls_dict = {
        "name": "hello",
        "type": "generic_control",
        "min": 0,
        "max": 1,
        "perturbation_magnitude": 0.01,
        "variables": [
            {
                "name": "var1",
                "initial_guess": 0.6,
            },
            {
                "name": "var2",
                "initial_guess": 0.6,
            },
        ],
    }

    parsed_config = ControlConfig(**controls_dict)
    assert isinstance(parsed_config.variables, list)

    [v1, v2] = parsed_config.variables

    assert isinstance(v1, ControlVariableConfig)
    assert isinstance(v2, ControlVariableConfig)

    assert v1.name == "var1"
    assert v2.name == "var2"


def test_that_optimization_config_is_initialized_with_cvar_config():
    optimization_dict = {
        "algorithm": "mesh_adaptive_search",
        "cvar": {"number_of_realizations": 999999},
    }

    parsed_config = OptimizationConfig(**optimization_dict)
    cvar_config = parsed_config.cvar

    assert isinstance(cvar_config, CVaRConfig)

    assert cvar_config.number_of_realizations == 999999
    assert "percentile" not in cvar_config


def test_that_get_output_dir_returns_same_for_old_and_new():
    config_src = {
        "wells": [
            {"name": "w00"},
        ],
        "controls": [
            {
                "name": "group_0",
                "type": "well_control",
                "min": 0,
                "max": 0.1,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "w00", "initial_guess": 0.0626},
                ],
            }
        ],
        "objective_functions": [{"name": "npv_function"}],
        "optimization": {
            "algorithm": "optpp_q_newton",
            "max_iterations": 2,
            "max_function_evaluations": 2,
            "perturbation_num": 2,
        },
        "model": {"realizations": [0, 1]},
        "environment": {
            "output_folder": "everezt_output",
            "simulation_folder": "tutorial_simulations",
            "random_seed": 999,
        },
    }

    config = EverestConfig.with_defaults(**config_src)

    assert Path(config.output_dir) == Path(config_src["environment"]["output_folder"])


def test_that_invalid_keys_are_linted():
    config_src = {
        "wells": [
            {"name": "w00"},
            {"naim": "w00", "dirll_date": ""},
        ],
        "welz": [],
        "controls": [
            {
                "name": "group_0",
                "type": "well_control",
                "inital_guss": "well_control",
                "min": 0,
                "max": 0.1,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "w00"},
                ],
            },
            {
                "name": "group_0",
                "type": "well_control",
                "initial_guess": "well_control",
                "min": 0,
                "max": 0.1,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "w00", "inital_guess": 0.0626},
                    {"name": "w01", "sampler": {"bakkend": "#DYSNEKTIC"}},
                ],
            },
        ],
        "controllers": [],
        "objective_functions": [{"allias": "ss", "name": "npv_function"}],
        "obejctive_fucctions": [],
        "optimisation": {
            "algorithm": "optpp_q_newton",
            "max_iterations": 2,
            "max_function_evaluations": 2,
            "perturbation_num": 2,
        },
        "optimization": {
            "allgorithm": "optpp_q_newton",
            "max_iterations": 2,
            "max_functions": 2,
            "perturbation_num": 2,
        },
        "model": {"Realizations": [0, 1], "reprot_setps,": []},
        "moddel": {"realizations": [0, 1]},
        "environment": {
            "output_folder": "everezt_output",
            "input_folder": "whatisthis",
            "simulation_folder": "tutorial_simulations",
            "random_seed": 999,
        },
        "envairånment": {
            "output_folder": "everezt_output",
            "simulation_folder": "tutorial_simulations",
            "random_seed": 999,
        },
        "export": {"dicsard_rejjecteded": "Tru"},
        "server": {"extrude_host": 49},
        "simulator": {"core_per_node": 49},
        "output_constraints": [{"name": "oc", "nam": 2, "target": 2}],
        "input_constraints": [{"nom": 3}],
        "install_data": [{"datta": "durr"}],
        "install_jobs": [{"jerb": "jebr"}],
        "install_workflow_jobs": [{"nm": "foo"}],
        "install_templates": [{"timplat": 2}],
        "config_path": "tmpz",
        "workflows": {"presimulation": ["job"]},
        "wrkflows": {"pre_simulation": ["job"]},
    }

    lint_result = EverestConfig.lint_config_dict(config_src)
    extra_errors_locs = [
        x["loc"] for x in lint_result if x["type"] == "extra_forbidden"
    ]

    assert set(extra_errors_locs) == {
        ("wells", 1, "dirll_date"),
        ("controls", 0, "inital_guss"),
        ("controls", 1, "variables", "list[ControlVariableConfig]", 0, "inital_guess"),
        (
            "controls",
            1,
            "variables",
            "list[ControlVariableGuessListConfig]",
            0,
            "inital_guess",
        ),
        (
            "controls",
            1,
            "variables",
            "list[ControlVariableConfig]",
            1,
            "sampler",
            "bakkend",
        ),
        (
            "controls",
            1,
            "variables",
            "list[ControlVariableGuessListConfig]",
            1,
            "sampler",
            "bakkend",
        ),
        ("objective_functions", 0, "allias"),
        ("optimization", "allgorithm"),
        ("optimization", "max_functions"),
        ("model", "Realizations"),
        ("model", "reprot_setps,"),
        ("environment", "input_folder"),
        ("input_constraints", 0, "nom"),
        ("output_constraints", 0, "nam"),
        ("install_jobs", 0, "jerb"),
        ("install_workflow_jobs", 0, "nm"),
        ("install_data", 0, "datta"),
        ("install_templates", 0, "timplat"),
        ("server", "extrude_host"),
        ("simulator", "core_per_node"),
        ("export", "dicsard_rejjecteded"),
        ("controllers",),
        ("envairånment",),
        ("moddel",),
        ("obejctive_fucctions",),
        ("optimisation",),
        ("wells", 1, "naim"),
        ("welz",),
        ("workflows", "presimulation"),
        ("wrkflows",),
    }


def test_that_log_level_property_is_consistent_with_environment_log_level():
    """
    Pydantic 1 somehow overrides property setters, and it was necessary
    to enforce this behavior by overriding __getattr__.
    This test verifies that the computed setter/getter works as intended
    """
    config_src = {
        "wells": [
            {
                "name": "dog",
            },
            {
                "name": "w01",
            },
        ],
        "controls": [
            {
                "name": "group_0",
                "type": "well_control",
                "min": 0,
                "max": 0.1,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "w01", "initial_guess": 0.0626},
                ],
            }
        ],
        "objective_functions": [{"name": "npv_function"}],
        "optimization": {
            "algorithm": "optpp_q_newton",
            "max_iterations": 2,
            "max_function_evaluations": 2,
            "perturbation_num": 2,
        },
        "model": {"realizations": [0, 1]},
        "environment": {
            "output_folder": "everezt_output",
            "simulation_folder": "tutorial_simulations",
            "log_level": "debug",
        },
    }

    levels = {
        "debug": logging.DEBUG,  # 10
        "info": logging.INFO,  # 20
        "warning": logging.WARNING,  # 30
        "error": logging.ERROR,  # 40
        "critical": logging.CRITICAL,  # 50
    }

    config = EverestConfig.with_defaults(**config_src)
    assert config.logging_level == levels[config.environment.log_level]

    for lvl_str, lvl_int in levels.items():
        config.environment.log_level = lvl_str
        assert config.logging_level == lvl_int


@pytest.mark.parametrize("config_class", [SimulatorConfig, ServerConfig])
@pytest.mark.parametrize("queue_system", ["lsf", "torque", "slurm", "local"])
def test_removed_queue_options_init(queue_system, config_class):
    config = {"queue_system": queue_system}
    with pytest.raises(ValueError, match=f"valid options for {queue_system} are"):
        config_class(**config)
