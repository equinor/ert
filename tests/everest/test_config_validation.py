import os
import pathlib
import re
from argparse import ArgumentParser
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ert.config import ConfigWarning
from ert.config.ert_config import ErtConfig
from ert.config.parsing import ConfigValidationError
from everest.config import EverestConfig, ModelConfig, ObjectiveFunctionConfig
from everest.config.control_variable_config import ControlVariableConfig
from everest.config.everest_config import EverestValidationError
from everest.config.sampler_config import SamplerConfig
from everest.simulator.everest_to_ert import (
    everest_to_ert_config_dict,
)


def all_errors(error: ValidationError, match: str):
    messages = [error_dict["msg"] for error_dict in error.errors()]
    instances = []
    for m in messages:
        instances.extend(re.findall(match, m))
    return instances


def test_that_sampler_config_with_wrong_method():
    with pytest.raises(ValueError, match=r"Sampler (.*) not found"):
        SamplerConfig(backend="scipy", method="hey")


def test_that_cvar_attrs_are_mutex():
    cvar = {"percentile": 0.1, "number_of_realizations": 3}
    with pytest.raises(ValueError, match="Invalid CVaR section"):
        EverestConfig.with_defaults(optimization={"cvar": cvar})


@pytest.mark.parametrize("nreals", [-1, 0, 8])
def test_that_cvar_nreals_interval_outside_range_errors(nreals):
    with pytest.raises(ValueError, match=f"number_of_realizations: \\(got {nreals}"):
        EverestConfig.with_defaults(
            optimization={
                "cvar": {
                    "number_of_realizations": nreals,
                }
            },
            model={"realizations": [1, 2, 3, 4, 5, 6]},
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
    with pytest.raises(ValueError, match=r".*greater than or equal to 0"):
        EverestConfig.with_defaults(simulator={"max_runtime": -1})

    EverestConfig.with_defaults(simulator={"max_runtime": 0})


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
    with pytest.raises(ValueError, match=r"(.*)`name` must be unique"):
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


def test_that_dot_not_in_control_names():
    with pytest.raises(ValueError, match=r"(.*)can not contain any dots"):
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


def test_that_scaled_range_is_valid_range():
    with pytest.raises(
        ValueError, match=r"(.*)must be a valid range \[a, b\], where a < b."
    ):
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
    with pytest.raises(ValueError, match=f"(.*)`{unique_key}` must be unique"):
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


def test_that_invalid_control_undefined_fields():
    with pytest.raises(
        ValueError,
        match=r"define min.* value.*define max*. value.*define initial_guess.* value",
    ):
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


def test_that_control_variables_index_is_defined_for_all_variables():
    with pytest.raises(
        ValueError, match="given either for all of the variables or for none of them"
    ):
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


def test_that_duplicate_output_constraint_names_raise_error():
    with pytest.raises(ValueError, match="Output constraint names must be unique"):
        EverestConfig.with_defaults(
            output_constraints=[
                {"target": 0.3, "name": "a"},
                {"target": 0.3, "name": "a"},
            ],
        )


@pytest.mark.parametrize(
    "constraint, expectation",
    [
        (
            {
                "name": "w110",
            },
            pytest.raises(
                ValueError, match="Must provide target or lower_bound/upper_bound"
            ),
        ),
        (
            {"name": "w110", "target": 1.0},
            does_not_raise(),
        ),
        (
            {"name": "w110", "lower_bound": 0.0, "upper_bound": 1.0},
            does_not_raise(),
        ),
        (
            {"name": "w110", "target": 1.0, "lower_bound": 0.0},
            pytest.raises(ValueError, match="Can not combine target and bounds"),
        ),
        (
            {"name": "w110", "target": 1.0, "upper_bound": 2.0},
            pytest.raises(ValueError, match="Can not combine target and bounds"),
        ),
    ],
)
def test_that_output_constraints_bounds_are_mutex(constraint, expectation):
    with expectation:
        EverestConfig.with_defaults(output_constraints=[constraint])


def test_that_variable_name_does_not_contain_dots():
    with pytest.raises(ValueError, match=r"(.*)can not contain any dots"):
        ControlVariableConfig(name="invalid.name")


@pytest.mark.parametrize(
    ["index_val", "expected_error"], [(0, False), (-1, True), (1, False)]
)
def test_that_variable_index_is_non_negative(index_val, expected_error):
    if expected_error:
        with pytest.raises(
            ValueError, match=r"(.*)Input should be greater than or equal to 0"
        ):
            ControlVariableConfig(name="var", index=index_val)
    else:
        ControlVariableConfig(name="var", index=index_val)


@pytest.mark.parametrize(
    ["perturbation", "expected_error"], [(0.0, True), (-1.0, True), (0.1, False)]
)
def test_that_variable_perturbation_is_positive(perturbation, expected_error):
    if expected_error:
        with pytest.raises(ValueError, match=r"(.*)Input should be greater than 0"):
            ControlVariableConfig(name="var", perturbation_magnitude=perturbation)
    else:
        ControlVariableConfig(name="var", perturbation_magnitude=perturbation)


def test_that_model_realizations_accept_only_positive_ints():
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        EverestConfig.with_defaults(model={"realizations": [-1, 1, 2, 3]})

    EverestConfig.with_defaults(model={"realizations": [0, 1, 2, 3]})


def test_that_model_realizations_weights_must_correspond_to_realizations():
    with pytest.raises(
        ValueError, match="Specified realizations_weights must have one weight per"
    ):
        EverestConfig.with_defaults(
            model={"realizations": [1, 2, 3], "realizations_weights": [1, 2]}
        )

    with pytest.raises(
        ValueError, match="Specified realizations_weights must have one weight per"
    ):
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
                "realizations_weights": [1, 2, 3, 4],
            }
        )

    EverestConfig.with_defaults(model={"realizations": [1, 2, 3]})
    EverestConfig.with_defaults(
        model={"realizations": [1, 2, 3], "realizations_weights": [5, 5, -5]}
    )


def test_that_missing_optimization_algorithm_errors():
    with pytest.raises(ValueError, match="Optimizer algorithm 'ddlygldt' not found"):
        EverestConfig.with_defaults(optimization={"algorithm": "ddlygldt"})


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
    with pytest.raises(ValueError, match=r"(.*)Input should be greater than 0"):
        EverestConfig.with_defaults(optimization={optimizer_attr: -1})

    with pytest.raises(ValueError, match=r"(.*)Input should be greater than 0"):
        EverestConfig.with_defaults(optimization={optimizer_attr: 0})

    EverestConfig.with_defaults(optimization={optimizer_attr: 1})


def test_that_min_realizations_success_is_nonnegative():
    with pytest.raises(
        ValueError, match=r"(.*)Input should be greater than or equal to 0"
    ):
        EverestConfig.with_defaults(optimization={"min_realizations_success": -1})

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
        model={"realizations": [0]},
    )

    for install_data_config in config.install_data:
        assert install_data_config.target == Path(data["source"]).name


@pytest.mark.parametrize(
    "install_data, expected_error_msg",
    [
        (
            {"source": "/", "link": True, "target": "bar.json"},
            "'/' is a mount point and can't be handled",
        ),
        (
            {"source": "baz/", "link": True, "target": "bar.json"},
            "No such file or directory",
        ),
        (
            {"source": None, "link": True, "target": "bar.json"},
            "Input should be a valid string",
        ),
        (
            {"source": "", "link": "false", "target": "bar.json"},
            " false could not be parsed to a boolean",
        ),
        (
            {"source": "baz/", "link": True, "target": 3},
            "Input should be a valid string",
        ),
    ],
)
def test_install_data_with_invalid_templates(
    tmp_path, install_data, expected_error_msg
):
    """
    Checks for InstallDataConfig's validations instantiating EverestConfig to also
    check invalid template rendering (e.g 'r{{ foo }}/) that maps to '/'
    """
    with pytest.raises(ValidationError) as exc_info:
        EverestConfig.with_defaults(
            controls=[
                {
                    "name": "initial_control",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "variables": [
                        {
                            "name": "param_a",
                            "initial_guess": 0.5,
                        }
                    ],
                }
            ],
            environment={"output_folder": str(tmp_path / "output")},
            model={"realizations": [1]},
            install_data=[install_data],
        )

    assert expected_error_msg in str(exc_info.value)


def test_that_install_data_source_exists(change_to_tmpdir):
    data = {
        "source": "relative/path",
        "target": "xxx",
    }
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(ValueError, match="No such file or directory"):
        EverestConfig.with_defaults(
            install_data=[data],
            config_path=Path("config_dir/test.yml"),
        )

    os.makedirs("config_dir/relative/path")
    EverestConfig.with_defaults(
        install_data=[data],
        config_path=Path("config_dir/test.yml"),
    )


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_non_existing_install_job_errors_deprecated(
    install_keyword, change_to_tmpdir
):
    os.makedirs("config_dir")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")
    with pytest.warns(
        ConfigWarning, match=f"`{install_keyword}: source` is deprecated"
    ):
        config = EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
            },
            config_path=Path("config_dir/test.yml"),
            **{install_keyword: [{"name": "test", "source": "non_existing"}]},
        )

    with pytest.raises(ConfigValidationError, match="No such file or directory:"):
        dictionary = everest_to_ert_config_dict(config)
        ErtConfig.from_dict(dictionary)


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_malformed_executable_errors_deprecated(
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

    with pytest.warns(
        ConfigWarning, match=f"`{install_keyword}: source` is deprecated"
    ):
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
        dictionary = everest_to_ert_config_dict(config)
        ErtConfig.from_dict(dictionary)


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_non_executable_executable_errors_deprecated(
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

    with pytest.warns(
        ConfigWarning, match=f"`{install_keyword}: source` is deprecated"
    ):
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
        dictionary = everest_to_ert_config_dict(config)
        ErtConfig.from_dict(dictionary)


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
    with open("non_executable", "w+", encoding="utf-8") as f:
        f.write("bla")

    os.chmod("non_executable", os.stat("non_executable").st_mode & ~0o111)
    assert not os.access("non_executable", os.X_OK)

    with pytest.raises(ValidationError, match="File not executable"):
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
            },
            config_path=Path("."),
            **{
                install_keyword: [
                    {"name": "test", "executable": "non_executable"},
                ]
            },
        )


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_existing_install_job_with_non_existing_executable_errors_deprecated(
    install_keyword, change_to_tmpdir
):
    with open("exec.ert", "w+", encoding="utf-8") as f:
        f.write(
            """EXECUTABLE non_existing
        """
        )

    assert not os.access("non_executable", os.X_OK)

    with pytest.warns(
        ConfigWarning, match=f"`{install_keyword}: source` is deprecated"
    ):
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
        dictionary = everest_to_ert_config_dict(config)
        ErtConfig.from_dict(dictionary)


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
    assert not os.access("non_executable", os.X_OK)

    with pytest.raises(ValidationError, match="Could not find executable"):
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
            },
            config_path=Path("."),
            **{
                install_keyword: [
                    {"name": "test", "executable": "non_executable"},
                ]
            },
        )


@pytest.mark.parametrize(
    ["value", "expect_error"],
    [
        (0.0, True),
        (-1.0, True),
        (0.1, False),
    ],
)
@pytest.mark.filterwarnings("ignore:normalization key is deprecated")
def test_that_objective_function_attrs_are_valid(value, expect_error):
    if expect_error:
        with pytest.raises(
            ValueError,
            match="The objective weight should be greater than 0",
        ):
            EverestConfig.with_defaults(
                objective_functions=[{"name": "npv", "weight": value}]
            )
    else:
        EverestConfig.with_defaults(
            objective_functions=[{"name": "npv", "weight": value}]
        )


def test_that_objective_function_weight_defined_for_all_or_no_function():
    with pytest.raises(
        ValueError, match=r"(.*) either for all of the objectives or for none of them"
    ):
        EverestConfig.with_defaults(
            objective_functions=[
                {"name": "npv", "weight": 0.7},
                {"name": "npv2"},
            ]
        )

    EverestConfig.with_defaults(
        objective_functions=[
            {"name": "npv"},
            {"name": "npv2"},
        ]
    )

    EverestConfig.with_defaults(
        objective_functions=[
            {"name": "npv", "weight": 0.7},
            {"name": "npv2", "weight": 0.3},
        ]
    )


@pytest.mark.parametrize(
    ["values", "expect_error"],
    [
        ([0.0, 0.0], True),
        ([0.0, -1.0], True),
        ([1.0, 0.0], False),
        ([1.0, -0.5], False),
    ],
)
def test_that_objective_function_weights_sum_is_positive(values, expect_error):
    if expect_error:
        with pytest.raises(
            ValueError,
            match="The sum of the objective weights should be greater than 0",
        ):
            EverestConfig.with_defaults(
                objective_functions=[
                    {"name": "npv", "weight": values[0]},
                    {"name": "npv2", "weight": values[1]},
                ]
            )
    else:
        EverestConfig.with_defaults(
            objective_functions=[
                {"name": "npv", "weight": values[0]},
                {"name": "npv2", "weight": values[1]},
            ]
        )


def test_that_install_templates_must_have_unique_names(change_to_tmpdir):
    for f in ["hey", "hesy", "heyyy"]:
        pathlib.Path(f).write_text(f, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Install_templates output_files "
        "must be unique. (.*) outputf \\(2 occurrences\\)",
    ):
        EverestConfig.with_defaults(
            install_templates=[
                {"template": "heyyy", "output_file": "outputf"},
                {"template": "hey", "output_file": "outputf"},
            ]
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
    with pytest.raises(ValueError, match=r"No.*file.*hello.*No.*file.*hey"):
        EverestConfig.with_defaults(
            install_templates=[
                {"template": "hello", "output_file": "output"},
                {"template": "hey", "output_file": "outputf"},
            ],
            config_path=Path("config_dir/test.yml"),
        )


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
    with pytest.raises(ValidationError, match="unknown workflow job job1"):
        EverestConfig.with_defaults(
            workflows={
                "pre_simulation": [
                    "job1 -i out -o result",
                ]
            },
        )


def test_deprecated_keyword_report_steps():
    with pytest.warns(ConfigWarning, match="report_steps .* can be removed"):
        ModelConfig(realizations=[0], report_steps=[])


def test_load_file_non_existing():
    with pytest.raises(FileNotFoundError):
        EverestConfig.load_file("non_existing.yml")


@pytest.mark.usefixtures("change_to_tmpdir")
def test_load_file_with_errors(capsys):
    content = dedent("""
    controls:
    -   initial_guess: 0.1
        max: not_a number
        min: -1.0
        name: point
        type: yolo_control
        variables:
        -   name: x
    install_jobs:
    -   executable: jobs/distance3.py
        name: distance
        invalid: invalid3
    model:
        realizations:
        - 0
    objective_functions:
    -   name: distance
""")

    with open("config_minimal_error.yml", "w", encoding="utf-8") as file:
        file.write(content)

    with pytest.raises(SystemExit):
        parser = ArgumentParser(prog="test")
        EverestConfig.load_file_with_argparser("config_minimal_error.yml", parser)

    captured = capsys.readouterr()

    assert "Found 3 validation errors" in captured.err
    assert "line: 7, column: 11" in captured.err
    assert (
        "Input should be 'well_control' or 'generic_control' (type=literal_error)"
        in captured.err
    )

    assert "line: 4, column: 10" in captured.err
    assert (
        "Input should be a valid number, "
        "unable to parse string as a number (type=float_parsing)"
    ) in captured.err

    assert "line: 13, column: 14" in captured.err
    assert "install_jobs -> 0 -> invalid" in captured.err
    assert "Extra inputs are not permitted (type=extra_forbidden)" in captured.err


@pytest.mark.parametrize(
    ["controls", "objectives", "error_msg"],
    [
        (
            [],
            [],
            [
                "controls\n  List should have at least 1 item after validation, not 0",
                (
                    "objective_functions\n  "
                    "List should have at least 1 item after validation, not 0"
                ),
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
    with pytest.raises(
        ValueError, match=r"normalization is deprecated .* replaced with scale"
    ):
        ObjectiveFunctionConfig(name="test", normalization=10)


def test_deprecated_objective_function_auto_normalize():
    with pytest.raises(
        ValueError,
        match=r"auto_normalize is deprecated .* replaced with auto_scale",
    ):
        ObjectiveFunctionConfig(name="test", auto_normalize=True)


def test_load_file_undefined_substitutions(min_config, change_to_tmpdir, capsys):
    config = min_config
    config["install_data"] = [
        {
            "source": "r{{configpath}}/../model/file.txt",
            "target": "r{{undefined_key }}/run_path",
        }
    ]

    with open("config.yml", mode="w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with pytest.raises(SystemExit):
        parser = ArgumentParser(prog="test")
        EverestConfig.load_file_with_argparser("config.yml", parser)

    captured = capsys.readouterr()
    assert (
        "Loading config file <config.yml> failed with: "
        "The following key is missing: ['r{{undefined_key }}']"
    ) in captured.err


@pytest.mark.parametrize(
    "key, value",
    [
        ["csv_output_filepath", "something"],
        ["csv_output_filepath", ""],
        ["csv_output_filepath", None],
        ["discard_gradient", True],
        ["discard_gradient", None],
        ["discard_gradient", "None"],
        ["discard_rejected", True],
        ["discard_rejected", None],
        ["discard_rejected", "None"],
        ["skip_export", True],
        ["skip_export", None],
        ["skip_export", "None"],
        ["batches", [0]],
        ["batches", []],
        ["batches", None],
        ["batches", "None"],
    ],
)
def test_export_deprecated_keys(key, value, min_config, change_to_tmpdir):
    config = min_config
    config["export"] = {key: value}

    with open("config.yml", mode="w", encoding="utf-8") as f:
        yaml.dump(config, f)

    parser = ArgumentParser(prog="test")
    match_msg = (
        f"'{key}' key is deprecated. You can safely remove it from the config file"
    )
    with pytest.warns(ConfigWarning, match=match_msg):
        EverestConfig.load_file_with_argparser("config.yml", parser)


@pytest.mark.parametrize(
    "realizations", ["1,", ",1", "1,,2", "1-", "-1", "2-1", "1--2", "1-,-2"]
)
def test_that_model_realizations_specs_are_invalid(realizations):
    with pytest.raises(
        ValueError, match=f"Invalid realizations specification: {realizations}"
    ):
        EverestConfig.with_defaults(model={"realizations": realizations})


@pytest.mark.parametrize(
    ("realizations", "expected"),
    [
        ("1", [1]),
        ("1,2", [1, 2]),
        ("1, 2", [1, 2]),
        ("1, 2, 2", [1, 2]),
        (" 1, 2 ", [1, 2]),
        ("1-1", [1]),
        ("1-3", [1, 2, 3]),
        ("1 -3", [1, 2, 3]),
        ("1- 3", [1, 2, 3]),
        ("1 - 3", [1, 2, 3]),
        ("1, 2-3", [1, 2, 3]),
        ("1, 3-4", [1, 3, 4]),
        ("1, 3-4, 2", [1, 2, 3, 4]),
        ("1, 3-4, 2", [1, 2, 3, 4]),
        ("6-6, 1, 3-4, 1, 2-5", [1, 2, 3, 4, 5, 6]),
    ],
)
def test_that_model_realizations_specs_are_valid(realizations, expected):
    config = EverestConfig.with_defaults(model={"realizations": realizations})
    assert config.model.realizations == expected


def test_that_nested_extra_types_are_validated_correctly(change_to_tmpdir):
    Path("everest_config.yml").write_text(
        dedent("""
        objective_functions:
          - name: func_name

        controls:
          - name: control_name
            type: generic_control
            min: 0
            max: 1
            initial_guess: 0.5
            variables:
                - name: var_name

        model:
          realizations: [0, 1]

        foo:
          bar:
            foobar: 44

    """),
        encoding="utf-8",
    )

    with pytest.raises(EverestValidationError) as err:
        EverestConfig.load_file("everest_config.yml")

    assert "ctx" in err.value.errors[0]
    assert err.value.errors[0]["ctx"] == {"line_number": 17}
    assert err.value.errors[0]["type"] == "extra_forbidden"
