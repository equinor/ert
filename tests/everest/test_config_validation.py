import os
import pathlib
import re
import warnings
from argparse import ArgumentParser
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ert.config import ConfigWarning
from ert.config.parsing import ConfigValidationError
from everest.config import EverestConfig, ModelConfig, ObjectiveFunctionConfig
from everest.config.control_variable_config import ControlVariableConfig
from everest.config.sampler_config import SamplerConfig
from everest.simulator.everest_to_ert import everest_to_ert_config
from tests.everest.utils import skipif_no_everest_models


def has_error(error: ValidationError | list[dict], match: str):
    messages = (
        [error_dict["msg"] for error_dict in error.errors()]
        if isinstance(error, ValidationError)
        else [e["msg"] for e in error]
    )
    pattern = re.compile(f"(.*){match}")
    return any(re.match(pattern, m) for m in messages)


def all_errors(error: ValidationError, match: str):
    messages = [error_dict["msg"] for error_dict in error.errors()]
    instances = []
    for m in messages:
        instances.extend(re.findall(match, m))
    return instances


def test_that_sampler_config_with_wrong_method():
    with pytest.raises(ValueError) as e:
        SamplerConfig(backend="scipy", method="hey")

    assert has_error(e.value, match="Sampler (.*) not found")


def test_that_duplicate_well_names_raise_error():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            wells=[
                {"name": "w110"},
                {"name": "w08"},
                {"name": "w10"},
                {"name": "w10"},
                {"name": "w09"},
                {"name": "w00"},
                {"name": "w01"},
                {"name": "w01"},
                {"name": "w01"},
            ],
        )

    assert has_error(e.value, match="Well names must be unique")


def test_that_dot_in_well_name_raises_error():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            wells=[
                {"name": "w.110"},
                {"name": "w.08"},
            ]
        )

    assert has_error(
        e.value,
        match="(.*)can not contain any dots",
    )


def test_that_negative_drill_time_raises_error():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            wells=[
                {"name": "w110", "drill_time": -1},
            ]
        )

    assert has_error(
        e.value,
        match="(.*)must be a positive number",
    )


def test_that_cvar_attrs_are_mutex():
    cvar = {"percentile": 0.1, "number_of_realizations": 3}
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(optimization={"cvar": cvar})

    assert has_error(e.value, match="Invalid CVaR section")


@pytest.mark.parametrize("nreals", [-1, 0, 8])
def test_that_cvar_nreals_interval_outside_range_errors(nreals):
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            optimization={
                "cvar": {
                    "number_of_realizations": nreals,
                }
            },
            model={"realizations": [1, 2, 3, 4, 5, 6]},
        )

    assert has_error(
        e.value,
        match=f"number_of_realizations: \\(got {nreals}",
    )


@pytest.mark.parametrize("nreals", [1, 2, 3, 4, 5])
def test_that_cvar_nreals_valid_doesnt_error(nreals):
    EverestConfig.with_defaults(
        optimization={
            "cvar": {
                "number_of_realizations": nreals,
            }
        },
        model={"realizations": [1, 2, 3, 4, 5, 6]},
    )


def test_that_max_runtime_errors_only_on_negative():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(simulator={"max_runtime": -1})

    EverestConfig.with_defaults(simulator={"max_runtime": 0})

    assert has_error(e.value, match=".*greater than or equal to 0")


def test_that_invalid_queue_system_errors():
    with pytest.raises(
        ValueError, match=r"does not match .*'local'.*'lsf'.*'slurm'.*'torque'"
    ):
        EverestConfig.with_defaults(simulator={"queue_system": {"name": "docal"}})


@pytest.mark.parametrize(
    ["cores", "expected_error"], [(0, False), (-1, True), (1, False)]
)
def test_that_cores_errors_only_on_lt_eq0(cores, expected_error):
    expectation = (
        pytest.raises(ValueError, match="greater than or equal to 0")
        if expected_error
        else does_not_raise()
    )
    with expectation:
        EverestConfig.with_defaults(
            simulator={"queue_system": {"name": "local", "max_running": cores}}
        )


@pytest.mark.parametrize(
    ["cores", "expected_error"], [(0, True), (-1, True), (1, False)]
)
def test_that_cores_per_node_errors_only_on_lt0(cores, expected_error):
    expectation = (
        pytest.raises(ValueError, match="greater than 0")
        if expected_error
        else does_not_raise()
    )
    with expectation:
        EverestConfig.with_defaults(simulator={"cores_per_node": cores})


def test_that_duplicate_control_names_raise_error():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "variables": [
                        {"name": "w00", "initial_guess": 0.06},
                    ],
                },
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "variables": [
                        {"name": "w01", "initial_guess": 0.09},
                    ],
                },
            ],
        )

    assert has_error(e.value, match="(.*)`name` must be unique")


def test_that_dot_not_in_control_names():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0.2",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "variables": [
                        {"name": "w00", "initial_guess": 0.06},
                        {"name": "w01", "initial_guess": 0.09},
                    ],
                }
            ]
        )

    assert has_error(
        e.value,
        match="(.*)can not contain any dots",
    )


def test_that_scaled_range_is_valid_range():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "scaled_range": [2, 1],
                    "variables": [
                        {"name": "w00", "initial_guess": 0.06},
                        {"name": "w01", "initial_guess": 0.09},
                    ],
                }
            ]
        )

    assert has_error(
        e.value,
        match=r"(.*)must be a valid range \[a, b\], where a < b.",
    )


@pytest.mark.parametrize(
    "variables, count",
    (
        pytest.param(
            [
                {  # upper bound (max)
                    "name": "w00",
                    "min": 0,
                    "max": 0.1,
                    "initial_guess": 1.09,
                },
                {  # lower bound (min)
                    "name": "w01",
                    "min": 0.5,
                    "max": 1,
                    "initial_guess": 0.29,
                },
            ],
            2,
            id="value",
        ),
        pytest.param(
            [
                {
                    "name": "w00",
                    "min": 0,
                    "max": 0.1,
                    "initial_guess": [1.09, 0.29],
                },
            ],
            1,
            id="vector",
        ),
    ),
)
def test_that_invalid_control_initial_guess_outside_bounds(
    variables: list[dict[str, Any]], count: int
):
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {"name": "group_0", "type": "well_control", "variables": variables}
            ]
        )

    assert (
        len(
            all_errors(
                e.value,
                match=r"must respect \d\.\d <= initial_guess <= \d\.\d",
            )
        )
        == count
    )


@pytest.mark.parametrize(
    "variables, unique_key",
    (
        pytest.param(
            [
                {"name": "w00", "initial_guess": 0.05},
                {"name": "w00", "initial_guess": 0.09},
            ],
            "name-index",
            id="name no index",
        ),
        pytest.param(
            [
                {"name": "w00", "index": 1, "initial_guess": 0.05},
                {"name": "w00", "index": 1, "initial_guess": 0.09},
            ],
            "name-index",
            id="name and index",
        ),
        pytest.param(
            [
                {"name": "w00", "initial_guess": [0.05, 0.09]},
                {"name": "w00", "initial_guess": [0.03, 0.07]},
            ],
            "name",
            id="vector",
        ),
    ),
)
def test_that_invalid_control_unique_entry(variables, unique_key):
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "max": 0,
                    "min": 0.1,
                    "variables": variables,
                }
            ]
        )

    assert has_error(
        e.value,
        match=f"(.*)`{unique_key}` must be unique",
    )


def test_that_invalid_control_undefined_fields():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "variables": [
                        {"name": "w00"},
                    ],
                }
            ]
        )

    for case in ["min", "max", "initial_guess"]:
        assert has_error(
            e.value,
            match=f"(.*)must define {case} value either at control level or variable",
        )


def test_that_control_variables_index_is_defined_for_all_variables():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "variables": [
                        {"name": "w01", "initial_guess": 0.06, "index": 0},
                        {"name": "w00", "initial_guess": 0.09},
                    ],
                }
            ]
        )

    assert has_error(
        e.value,
        match="(.*)given either for all of the variables or for none of them",
    )


def test_that_duplicate_output_constraint_names_raise_error():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            output_constraints=[
                {"target": 0.3, "name": "c110"},
                {"target": 0.3, "name": "c08"},
                {"target": 0.3, "name": "c10"},
                {"target": 0.3, "name": "c10"},
                {"target": 0.3, "name": "c09"},
                {"target": 0.3, "name": "c00"},
                {"target": 0.3, "name": "c01"},
                {"target": 0.3, "name": "c01"},
                {"target": 0.3, "name": "c01"},
            ],
        )

    assert has_error(e.value, match="Output constraint names must be unique")


def test_that_output_constraints_bounds_are_mutex():
    output_constraint = {
        "name": "w110",
    }
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(output_constraints=[output_constraint])

    assert has_error(
        e.value, match="Output constraints must have only one of the following"
    )

    output_constraint["target"] = 1.0
    EverestConfig.with_defaults(output_constraints=[output_constraint])

    output_constraint["upper_bound"] = 2.0
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(output_constraints=[output_constraint])

    assert has_error(
        e.value, match="Output constraints must have only one of the following"
    )

    output_constraint["lower_bound"] = 0.5
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(output_constraints=[output_constraint])

    assert has_error(
        e.value, match="Output constraints must have only one of the following"
    )

    del output_constraint["upper_bound"]
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(output_constraints=[output_constraint])

    assert has_error(
        e.value, match="Output constraints must have only one of the following"
    )
    del output_constraint["target"]
    EverestConfig.with_defaults(output_constraints=[output_constraint])

    del output_constraint["lower_bound"]
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(output_constraints=[output_constraint])

    assert has_error(
        e.value, match="Output constraints must have only one of the following"
    )


def test_that_variable_name_does_not_contain_dots():
    with pytest.raises(ValueError) as e:
        ControlVariableConfig(name="invalid.name")
    assert has_error(e.value, match="(.*)can not contain any dots")


@pytest.mark.parametrize(
    ["index_val", "expected_error"], [(0, False), (-1, True), (1, False)]
)
def test_that_variable_index_is_non_negative(index_val, expected_error):
    if expected_error:
        with pytest.raises(ValueError) as e:
            ControlVariableConfig(name="var", index=index_val)
        assert has_error(
            e.value, match="(.*)Input should be greater than or equal to 0"
        )
    else:
        ControlVariableConfig(name="var", index=index_val)


@pytest.mark.parametrize(
    ["perturbation", "expected_error"], [(0.0, True), (-1.0, True), (0.1, False)]
)
def test_that_variable_perturbation_is_positive(perturbation, expected_error):
    if expected_error:
        with pytest.raises(ValueError) as e:
            ControlVariableConfig(name="var", perturbation_magnitude=perturbation)
        assert has_error(e.value, match="(.*)Input should be greater than 0")
    else:
        ControlVariableConfig(name="var", perturbation_magnitude=perturbation)


def test_that_model_realizations_accept_only_positive_ints():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(model={"realizations": [-1, 1, 2, 3]})

    assert has_error(e.value, match="Input should be greater than or equal to 0")

    EverestConfig.with_defaults(model={"realizations": [0, 1, 2, 3]})


def test_that_model_realizations_weights_must_correspond_to_realizations():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            model={"realizations": [1, 2, 3], "realizations_weights": [1, 2]}
        )
    assert has_error(
        e.value, match="Specified realizations_weights must have one weight per"
    )

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
                "realizations_weights": [1, 2, 3, 4],
            }
        )
    assert has_error(
        e.value, match="Specified realizations_weights must have one weight per"
    )

    EverestConfig.with_defaults(model={"realizations": [1, 2, 3]})
    EverestConfig.with_defaults(
        model={"realizations": [1, 2, 3], "realizations_weights": [5, 5, -5]}
    )


def test_that_missing_optimization_algorithm_errors():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(optimization={"algorithm": "ddlygldt"})

    assert has_error(e.value, match="Optimizer algorithm 'ddlygldt' not found")


@pytest.mark.parametrize(
    "optimizer_attr",
    [
        "perturbation_num",
        "max_iterations",
        "max_function_evaluations",
        "max_batch_num",
        "min_pert_success",
    ],
)
def test_that_some_optimization_attrs_must_be_positive(optimizer_attr):
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(optimization={optimizer_attr: -1})

    assert has_error(e.value, match="(.*)Input should be greater than 0")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(optimization={optimizer_attr: 0})

    assert has_error(e.value, match="(.*)Input should be greater than 0")

    EverestConfig.with_defaults(optimization={optimizer_attr: 1})


def test_that_min_realizations_success_is_nonnegative():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(optimization={"min_realizations_success": -1})

    assert has_error(e.value, match="(.*)Input should be greater than or equal to 0")

    EverestConfig.with_defaults(optimization={"min_realizations_success": 0})


@pytest.mark.parametrize(
    ["target", "link"],
    [
        (".", True),
        ("./", True),
        (".", False),
        ("./", False),
    ],
)
def test_that_install_data_allows_runpath_root_as_target(
    target, link, change_to_tmpdir
):
    data = {"source": "relative/path_<GEO_ID>", "target": target, "link": link}
    os.makedirs("config_dir/relative/path_0")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")
    config = EverestConfig.with_defaults(
        install_data=[data],
        config_path=Path("config_dir/test.yml"),
        model=ModelConfig(realizations=[0]),
    )

    for install_data_config in config.install_data:
        assert install_data_config.target == Path(data["source"]).name


def test_that_install_data_source_exists(change_to_tmpdir):
    data = {
        "source": "relative/path",
        "target": "xxx",
    }
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            install_data=[data],
            config_path=Path("config_dir/test.yml"),
        )
    assert has_error(e.value, match="No such file or directory")

    os.makedirs("config_dir/relative/path")
    EverestConfig.with_defaults(
        install_data=[data],
        config_path=Path("config_dir/test.yml"),
    )


def test_that_model_data_file_exists(change_to_tmpdir):
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            model={"realizations": [1, 2, 3], "data_file": "relative/path"},
            config_path=Path("config_dir/test.yml"),
        )

    assert has_error(e.value, match="No such file or directory")

    os.makedirs("config_dir/relative/path")

    EverestConfig.with_defaults(
        model={"realizations": [1, 2, 3], "data_file": "relative/path"},
        config_path=Path("config_dir/test.yml"),
    )


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_non_existing_install_job_errors(install_keyword, change_to_tmpdir):
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")
    config = EverestConfig.with_defaults(
        model={
            "realizations": [1, 2, 3],
        },
        config_path=Path("config_dir/test.yml"),
        **{install_keyword: [{"name": "test", "source": "non_existing"}]},
    )

    with pytest.raises(ConfigValidationError, match="No such file or directory:"):
        everest_to_ert_config(config)


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_malformed_executable_errors(
    install_keyword, change_to_tmpdir
):
    with open("malformed.ert", "w+", encoding="utf-8") as f:
        f.write(
            """EXECUTABLE
        """
        )
    with open("malformed2.ert", "w+", encoding="utf-8") as f:
        f.write(
            """EXECUTABLE 1 two 3
               EXECUTABLE one two
        """
        )

    config = EverestConfig.with_defaults(
        model={
            "realizations": [1, 2, 3],
        },
        config_path=Path("."),
        **{
            install_keyword: [
                {"name": "test", "source": "malformed.ert"},
                {"name": "test2", "source": "malformed2.ert"},
            ]
        },
    )

    with pytest.raises(
        ConfigValidationError, match="EXECUTABLE must have at least 1 arguments"
    ):
        everest_to_ert_config(config)


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_non_executable_executable_errors(
    install_keyword, change_to_tmpdir
):
    with open("exec.ert", "w+", encoding="utf-8") as f:
        f.write(
            """EXECUTABLE non_executable
        """
        )

    with open("non_executable", "w+", encoding="utf-8") as f:
        f.write("bla")

    os.chmod("non_executable", os.stat("non_executable").st_mode & ~0o111)
    assert not os.access("non_executable", os.X_OK)

    config = EverestConfig.with_defaults(
        model={
            "realizations": [1, 2, 3],
        },
        config_path=Path("."),
        **{
            install_keyword: [
                {"name": "test", "source": "exec.ert"},
            ]
        },
    )

    with pytest.raises(ConfigValidationError, match="File not executable"):
        everest_to_ert_config(config)


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_non_existing_executable_errors(
    install_keyword, change_to_tmpdir
):
    with open("exec.ert", "w+", encoding="utf-8") as f:
        f.write(
            """EXECUTABLE non_existing
        """
        )

    assert not os.access("non_executable", os.X_OK)

    config = EverestConfig.with_defaults(
        model={
            "realizations": [1, 2, 3],
        },
        config_path=Path("."),
        **{
            install_keyword: [
                {"name": "test", "source": "exec.ert"},
            ]
        },
    )

    with pytest.raises(ConfigValidationError, match="Could not find executable"):
        everest_to_ert_config(config)


@pytest.mark.parametrize(
    ["key", "value", "expected_error"],
    [
        ("weight", 0.0, "(.*)Input should be greater than 0"),
        ("weight", -1.0, "(.*)Input should be greater than 0"),
        ("weight", 0.1, None),
        ("normalization", 0.0, "(.*) value cannot be zero"),
        ("normalization", -1.0, None),
        ("normalization", 0.1, None),
    ],
)
def test_that_objective_function_attrs_are_valid(key, value, expected_error):
    if expected_error:
        with pytest.raises(ValueError) as e:
            EverestConfig.with_defaults(
                objective_functions=[{"name": "npv", key: value}]
            )
        assert has_error(e.value, expected_error)
    else:
        EverestConfig.with_defaults(objective_functions=[{"name": "npv", key: value}])


def test_that_objective_function_weight_defined_for_all_or_no_function():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            objective_functions=[
                {"name": "npv", "weight": 0.7},
                {"name": "npv2"},
            ]
        )
    assert has_error(
        e.value, "(.*) either for all of the objectives or for none of them"
    )

    EverestConfig.with_defaults(
        objective_functions=[
            {"name": "npv", "weight": 0.7},
            {"name": "npv2", "weight": 0.3},
        ]
    )


def test_that_objective_function_aliases_are_consistent():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            objective_functions=[
                {"name": "npv"},
                {"name": "npv2", "alias": "bad_one"},
            ]
        )
    assert has_error(e.value, "Invalid alias (.*)")

    EverestConfig.with_defaults(
        objective_functions=[
            {"name": "npv"},
            {"name": "npv2", "alias": "npv"},
        ]
    )


def test_that_install_templates_must_have_unique_names(change_to_tmpdir):
    for f in ["hey", "hesy", "heyyy"]:
        pathlib.Path(f).write_text(f, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            install_templates=[
                {"template": "heyyy", "output_file": "outputf"},
                {"template": "hey", "output_file": "outputf"},
            ]
        )

    assert has_error(
        e.value,
        match="Install_templates output_files "
        "must be unique. (.*) outputf \\(2 occurrences\\)",
    )
    print("Install_templates templates must be unique")

    EverestConfig.with_defaults(
        install_templates=[
            {"template": "hey", "output_file": "outputf"},
            {"template": "hesy", "output_file": "outputff"},
        ]
    )


def test_that_install_template_template_must_be_existing_file(change_to_tmpdir):
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            install_templates=[
                {"template": "hello", "output_file": "output"},
                {"template": "hey", "output_file": "outputf"},
            ],
            config_path=Path("config_dir/test.yml"),
        )

    assert has_error(e.value, "No such file or directory.*hey")
    assert has_error(e.value, "No such file or directory.*hello")


def test_that_missing_required_fields_cause_error():
    with pytest.raises(ValidationError) as e:
        EverestConfig()

    error_dicts = e.value.errors()

    # Expect missing error for:
    # controls, objective_functions, config_path, model
    assert len(error_dicts) == 4

    config_with_defaults = EverestConfig.with_defaults()
    config_args = {}
    required_argnames = [
        "controls",
        "objective_functions",
        "config_path",
        "model",
    ]

    for key in required_argnames:
        with pytest.raises(ValidationError) as e:
            EverestConfig(**config_args)

        assert len(e.value.errors()) == len(required_argnames) - len(config_args)
        config_args[key] = getattr(config_with_defaults, key)


def test_that_non_existing_workflow_jobs_cause_error():
    with pytest.raises(ValidationError) as e:
        EverestConfig.with_defaults(
            install_workflow_jobs=[{"name": "job0", "source": "jobs/JOB"}],
            workflows={
                "pre_simulation": [
                    "job0 -i in -o out",
                    "job1 -i out -o result",
                ]
            },
        )
    assert has_error(e.value, "unknown workflow job job1")


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.parametrize(
    ["objective", "forward_model", "warning_msg"],
    [
        (
            ["rf"],
            ["well_trajectory -c Something -E Something", "rf -s TEST -o rf"],
            None,
        ),
        (
            ["npv", "rf"],
            ["rf -s TEST -o rf"],
            "Warning: Forward model might not write the required output file for \\['npv'\\]",
        ),
        (
            ["npv", "npv2"],
            ["rf -s TEST -o rf"],
            "Warning: Forward model might not write the required output files for \\['npv', 'npv2'\\]",
        ),
        (
            ["rf"],
            ["rf -s TEST -o rf"],
            None,
        ),
        (
            ["rf"],
            None,
            None,
        ),
    ],
)
def test_warning_forward_model_write_objectives(objective, forward_model, warning_msg):
    # model.realizations is non-empty and therefore this test will run full validation on forward model schema, we don't want that for this test
    with patch("everest.config.everest_config.validate_forward_model_configs"):
        if warning_msg is not None:
            with pytest.warns(ConfigWarning, match=warning_msg):
                EverestConfig.with_defaults(
                    objective_functions=[{"name": o} for o in objective],
                    forward_model=forward_model,
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=ConfigWarning)
                EverestConfig.with_defaults(
                    objective_functions=[{"name": o} for o in objective],
                    forward_model=forward_model,
                )


def test_deprecated_keyword_report_steps():
    with pytest.warns(ConfigWarning, match="report_steps .* can be removed"):
        ModelConfig(realizations=[0], report_steps=[])


def test_load_file_non_existing():
    with pytest.raises(FileNotFoundError):
        EverestConfig.load_file("non_existing.yml")


def test_load_file_with_errors(copy_math_func_test_data_to_tmp, capsys):
    with open("config_minimal.yml", encoding="utf-8") as file:
        content = file.read()

    with open("config_minimal_error.yml", "w", encoding="utf-8") as file:
        content = content.replace("generic_control", "yolo_control")
        content = content.replace("max: 1.0", "max: not_a number")
        pos = content.find("name: distance")
        content = content[: pos + 14] + "\n    invalid: invalid" + content[pos + 14 :]
        file.write(content)

    with pytest.raises(SystemExit):
        parser = ArgumentParser(prog="test")
        EverestConfig.load_file_with_argparser("config_minimal_error.yml", parser)

    captured = capsys.readouterr()

    assert "Found 3 validation error" in captured.err
    assert "line: 3, column: 11" in captured.err
    assert (
        "Input should be 'well_control' or 'generic_control' (type=literal_error)"
        in captured.err
    )

    assert "line: 5, column: 10" in captured.err
    assert (
        "Input should be a valid number, unable to parse string as a number (type=float_parsing)"
        in captured.err
    )

    assert "line: 16, column: 5" in captured.err
    assert "Extra inputs are not permitted (type=extra_forbidden)" in captured.err


@pytest.mark.parametrize(
    ["controls", "objectives", "error_msg"],
    [
        (
            [],
            [],
            [
                "controls\n  List should have at least 1 item after validation, not 0",
                "objective_functions\n  List should have at least 1 item after validation, not 0",
            ],
        ),
        (
            [],
            [{"name": "npv"}],
            [
                "controls\n  List should have at least 1 item after validation, not 0",
            ],
        ),
        (
            [
                {
                    "name": "test",
                    "type": "generic_control",
                    "initial_guess": 0.5,
                    "variables": [
                        {"name": "test", "min": 0, "max": 1},
                    ],
                }
            ],
            [{"name": "npv"}],
            [],
        ),
    ],
)
def test_warning_empty_controls_and_objectives(controls, objectives, error_msg):
    if error_msg:
        with pytest.raises(ValueError) as e:
            EverestConfig.with_defaults(
                objective_functions=objectives,
                controls=controls,
            )

        for msg in error_msg:
            assert msg in str(e.value)
    else:
        EverestConfig.with_defaults(
            objective_functions=objectives,
            controls=controls,
        )


def test_deprecated_objective_function_normalization():
    with pytest.warns(
        ConfigWarning, match="normalization key is deprecated .* replaced with scaling"
    ):
        ObjectiveFunctionConfig(name="test", normalization=10)


def test_deprecated_objective_function_auto_normalize():
    with pytest.warns(
        ConfigWarning,
        match="auto_normalize key is deprecated .* replaced with auto_scale",
    ):
        ObjectiveFunctionConfig(name="test", auto_normalize=True)


@pytest.mark.parametrize(
    "normalization, scaling, auto_normalize, auto_scale",
    [
        (None, None, None, None),
        (0.2, None, None, None),
        (0.42, 0.24, None, None),
        (None, 0.24, None, None),
        (None, None, True, None),
        (None, None, True, False),
        (None, None, None, False),
        (0.42, 0.24, True, False),
    ],
)
def test_objective_function_scaling_is_backward_compatible_with_scaling(
    normalization, auto_normalize, scaling, auto_scale
):
    o = ObjectiveFunctionConfig(
        name="test",
        normalization=normalization,
        auto_normalize=auto_normalize,
        scaling=scaling,
        auto_scale=auto_scale,
    )
    if scaling is None and normalization is not None:
        assert o.scaling == 1 / o.normalization
    else:
        assert o.scaling == scaling
        assert o.normalization == normalization

    if auto_scale is None and auto_normalize is not None:
        assert o.auto_scale == o.auto_normalize
    else:
        assert o.auto_scale == auto_scale
        assert o.auto_normalize == auto_normalize
