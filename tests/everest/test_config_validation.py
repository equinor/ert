import json
import os
import pathlib
import re
from argparse import ArgumentParser
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from textwrap import dedent
from typing import Any

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given
from pydantic import ValidationError

from ert.config import ConfigValidationError, ConfigWarning
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import (
    EverestConfig,
    ModelConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
)
from everest.config.control_variable_config import ControlVariableConfig
from everest.config.everest_config import EverestValidationError
from everest.config.forward_model_config import ForwardModelStepConfig
from everest.config.sampler_config import SamplerConfig
from everest.config.validation_utils import _OVERWRITE_MESSAGE, _RESERVED_WORDS
from tests.everest.utils import everest_config_with_defaults


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
        everest_config_with_defaults(optimization={"cvar": cvar})


@pytest.mark.parametrize("nreals", [-1, 0, 8])
def test_that_cvar_nreals_interval_outside_range_errors(nreals):
    with pytest.raises(ValueError, match=f"number_of_realizations: \\(got {nreals}"):
        everest_config_with_defaults(
            optimization={
                "cvar": {
                    "number_of_realizations": nreals,
                }
            },
            model={"realizations": [1, 2, 3, 4, 5, 6]},
        )


@pytest.mark.parametrize("nreals", [1, 2, 3, 4, 5])
def test_that_cvar_nreals_valid_doesnt_error(nreals):
    everest_config_with_defaults(
        optimization={
            "cvar": {
                "number_of_realizations": nreals,
            }
        },
        model={"realizations": [1, 2, 3, 4, 5, 6]},
    )


def test_that_max_runtime_errors_only_on_negative():
    with pytest.raises(ValueError, match=r".*greater than or equal to 0"):
        everest_config_with_defaults(simulator={"max_runtime": -1})

    everest_config_with_defaults(simulator={"max_runtime": 0})


def test_that_invalid_queue_system_errors():
    with pytest.raises(
        ValueError, match=r"does not match .*'local'.*'lsf'.*'slurm'.*'torque'"
    ):
        everest_config_with_defaults(simulator={"queue_system": {"name": "docal"}})


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
        everest_config_with_defaults(
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
        everest_config_with_defaults(simulator={"cores_per_node": cores})


def test_that_duplicate_control_names_raise_error():
    with pytest.raises(ValueError, match=r"(.*)`name` must be unique"):
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "perturbation_magnitude": 0.01,
                    "variables": [
                        {"name": "w00", "initial_guess": 0.06},
                    ],
                },
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "perturbation_magnitude": 0.01,
                    "variables": [
                        {"name": "w01", "initial_guess": 0.09},
                    ],
                },
            ],
        )


def test_that_dot_not_in_control_names():
    with pytest.raises(ValueError, match=r"(.*)can not contain any dots"):
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0.2",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "perturbation_magnitude": 0.01,
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
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "perturbation_magnitude": 0.01,
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
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "perturbation_magnitude": 0.01,
                    "variables": variables,
                }
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
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "max": 0,
                    "min": 0.1,
                    "perturbation_magnitude": 0.01,
                    "variables": variables,
                }
            ]
        )


def test_that_invalid_control_undefined_fields():
    with pytest.raises(
        ValueError,
        match=r"define min.* value.*define max*. value.*define initial_guess.* value",
    ):
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "perturbation_magnitude": 0.01,
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
        everest_config_with_defaults(
            controls=[
                {
                    "name": "group_0",
                    "type": "well_control",
                    "min": 0,
                    "max": 0.1,
                    "perturbation_magnitude": 0.01,
                    "variables": [
                        {"name": "w01", "initial_guess": 0.06, "index": 0},
                        {"name": "w00", "initial_guess": 0.09},
                    ],
                }
            ]
        )


def test_that_duplicate_output_constraint_names_raise_error():
    with pytest.raises(ValueError, match="Output constraint names must be unique"):
        everest_config_with_defaults(
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
        everest_config_with_defaults(output_constraints=[constraint])


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
        everest_config_with_defaults(model={"realizations": [-1, 1, 2, 3]})

    everest_config_with_defaults(model={"realizations": [0, 1, 2, 3]})


def test_that_model_realizations_weights_must_correspond_to_realizations():
    with pytest.raises(
        ValueError, match="Specified realizations_weights must have one weight per"
    ):
        everest_config_with_defaults(
            model={"realizations": [1, 2, 3], "realizations_weights": [1, 2]}
        )

    with pytest.raises(
        ValueError, match="Specified realizations_weights must have one weight per"
    ):
        everest_config_with_defaults(
            model={
                "realizations": [1, 2, 3],
                "realizations_weights": [1, 2, 3, 4],
            }
        )

    everest_config_with_defaults(model={"realizations": [1, 2, 3]})
    everest_config_with_defaults(
        model={"realizations": [1, 2, 3], "realizations_weights": [5, 5, -5]}
    )


def test_that_missing_optimization_algorithm_errors():
    with pytest.raises(ValueError, match="Optimizer algorithm 'ddlygldt' not found"):
        everest_config_with_defaults(optimization={"algorithm": "ddlygldt"})


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
        everest_config_with_defaults(optimization={optimizer_attr: -1})

    with pytest.raises(ValueError, match=r"(.*)Input should be greater than 0"):
        everest_config_with_defaults(optimization={optimizer_attr: 0})

    everest_config_with_defaults(optimization={optimizer_attr: 1})


def test_that_min_realizations_success_is_nonnegative():
    with pytest.raises(
        ValueError, match=r"(.*)Input should be greater than or equal to 0"
    ):
        everest_config_with_defaults(optimization={"min_realizations_success": -1})

    everest_config_with_defaults(optimization={"min_realizations_success": 0})


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
    Path("config_dir/relative/path_0").mkdir(parents=True)
    Path("config_dir/test.yml").write_text(" ", encoding="utf-8")
    config = everest_config_with_defaults(
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
            "Either source or data must be provided",
        ),
        (
            {"source": "", "link": "false", "target": "bar.json"},
            " false could not be parsed to a boolean",
        ),
        (
            {"source": "baz/", "link": True, "target": 3},
            "Input should be a valid string",
        ),
        (
            {"link": True, "target": "bar.json"},
            "Either source or data must be provided",
        ),
        (
            {"source": "baz/", "data": {"foo": 1}, "link": True, "target": "bar.json"},
            "The data and source options are mutually exclusive",
        ),
        (
            {"data": {"foo": 1}, "link": True, "target": "bar.txt"},
            "Invalid target extension .txt (.json expected).",
        ),
        (
            {"data": {"foo": 1}, "link": False, "target": ""},
            "A target name must be provided with data.",
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
        everest_config_with_defaults(
            controls=[
                {
                    "name": "initial_control",
                    "type": "well_control",
                    "min": 0,
                    "max": 1,
                    "perturbation_magnitude": 0.01,
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
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").write_text(" ", encoding="utf-8")

    with pytest.raises(ValueError, match="No such file or directory"):
        everest_config_with_defaults(
            install_data=[data],
            config_path=Path("config_dir/test.yml"),
        )

    Path("config_dir/relative/path").mkdir(parents=True)
    everest_config_with_defaults(
        install_data=[data],
        config_path=Path("config_dir/test.yml"),
    )


def test_that_repeated_install_elements_to_same_location_will_fail(
    change_to_tmpdir: None,
) -> None:
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").touch()
    Path("config_dir/foo").mkdir()
    with pytest.raises(ValueError, match=_OVERWRITE_MESSAGE):
        everest_config_with_defaults(
            install_data=[
                {"source": "foo", "target": "foo"},
                {"source": "foo", "target": "foo"},
            ],
            config_path=Path("config_dir/test.yml"),
        )


def test_that_install_elements_cannot_install_over_previously_installed_folder(
    change_to_tmpdir: None,
) -> None:
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").touch()
    Path("config_dir/foo/bar").mkdir(parents=True)
    with pytest.raises(ValueError, match=_OVERWRITE_MESSAGE):
        everest_config_with_defaults(
            install_data=[
                {"source": "foo/bar", "target": "foo/bar"},
                {"source": "foo", "target": "foo"},
            ],
            config_path=Path("config_dir/test.yml"),
        )


def test_that_install_elements_cannot_install_into_previously_installed_folders(
    change_to_tmpdir: None,
) -> None:
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").touch()
    Path("config_dir/foo").mkdir()
    Path("config_dir/bar").touch()
    with pytest.raises(ValueError, match=_OVERWRITE_MESSAGE):
        everest_config_with_defaults(
            install_data=[
                {"source": "foo", "target": "foo"},
                {"source": "bar", "target": "foo/bar"},
            ],
            config_path=Path("config_dir/test.yml"),
        )


@pytest.mark.integration_test
def test_that_install_data_with_inline_data_generates_a_file(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {
            "algorithm": "optpp_q_newton",
            "perturbation_num": 1,
            "max_batch_num": 1,
        },
        "install_data": [
            {
                "data": {"x": 1},
                "target": "output.json",
            }
        ],
    }
    config = EverestConfig.model_validate(config_dict)
    runtime_plugins = get_site_plugins()
    run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)
    run_model.run_experiment(EvaluatorServerConfig())
    for expected_dir in ("evaluation_0", "perturbation_0"):
        expected_file = Path(
            f"everest_output/sim_output/batch_0/realization_0/{expected_dir}/output.json"
        )
        assert expected_file.exists()
        with expected_file.open(encoding="utf-8") as fp:
            data = json.load(fp)
        assert set(data.keys()) == {"x"}
        assert data["x"] == 1


@pytest.mark.parametrize(
    ["install_keyword"],
    [
        ("install_jobs",),
        ("install_workflow_jobs",),
    ],
)
def test_that_either_source_or_executable_is_provided(install_keyword):
    with pytest.raises(
        ValidationError, match="Either source or executable must be provided"
    ):
        everest_config_with_defaults(
            model={"realizations": [1, 2, 3]},
            config_path=Path("."),
            **{install_keyword: [{"name": "test"}]},
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
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").write_text(" ", encoding="utf-8")
    with pytest.raises(ValidationError, match="No such file or directory:"):
        everest_config_with_defaults(
            model={
                "realizations": [1, 2, 3],
            },
            config_path=Path("config_dir/test.yml"),
            **{install_keyword: [{"name": "test", "source": "non_existing"}]},
        )


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
        config = everest_config_with_defaults(
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
        EverestRunModel.create(config)


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
        config = everest_config_with_defaults(
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
        EverestRunModel.create(config)


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
        everest_config_with_defaults(
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
        config = everest_config_with_defaults(
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
        EverestRunModel.create(config)


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
        everest_config_with_defaults(
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
            everest_config_with_defaults(
                objective_functions=[{"name": "npv", "weight": value}]
            )
    else:
        everest_config_with_defaults(
            objective_functions=[{"name": "npv", "weight": value}]
        )


def test_that_objective_function_weight_defined_for_all_or_no_function():
    with pytest.raises(
        ValueError, match=r"(.*) either for all of the objectives or for none of them"
    ):
        everest_config_with_defaults(
            objective_functions=[
                {"name": "npv", "weight": 0.7},
                {"name": "npv2"},
            ]
        )

    everest_config_with_defaults(
        objective_functions=[
            {"name": "npv"},
            {"name": "npv2"},
        ]
    )

    everest_config_with_defaults(
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
            everest_config_with_defaults(
                objective_functions=[
                    {"name": "npv", "weight": values[0]},
                    {"name": "npv2", "weight": values[1]},
                ]
            )
    else:
        everest_config_with_defaults(
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
        everest_config_with_defaults(
            install_templates=[
                {"template": "heyyy", "output_file": "outputf"},
                {"template": "hey", "output_file": "outputf"},
            ]
        )

    print("Install_templates templates must be unique")

    everest_config_with_defaults(
        install_templates=[
            {"template": "hey", "output_file": "outputf"},
            {"template": "hesy", "output_file": "outputff"},
        ]
    )


def test_that_install_template_template_must_be_existing_file(change_to_tmpdir):
    Path("config_dir").mkdir()
    Path("config_dir/test.yml").write_text(" ", encoding="utf-8")
    with pytest.raises(ValueError, match=r"No.*file.*hello.*No.*file.*hey"):
        everest_config_with_defaults(
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

    config_with_defaults = everest_config_with_defaults()
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
        everest_config_with_defaults(
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
        perturbation_magnitude: 0.01
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

    Path("config_minimal_error.yml").write_text(content, encoding="utf-8")

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

    assert "line: 14, column: 14" in captured.err
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
                    "perturbation_magnitude": 0.01,
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
            everest_config_with_defaults(
                objective_functions=objectives,
                controls=controls,
            )

        for msg in error_msg:
            assert msg in str(e.value)
    else:
        everest_config_with_defaults(
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


def test_that_auto_scale_and_objective_scale_are_mutually_exclusive(tmp_path):
    with pytest.raises(
        ValueError,
        match=(
            "The auto_scale option in the optimization section and the scale "
            "options in the objective_functions section are mutually exclusive"
        ),
    ):
        everest_config_with_defaults(
            optimization=OptimizationConfig(auto_scale=True),
            objective_functions=[
                ObjectiveFunctionConfig(name=f"f{i:03d}", scale=1.0) for i in range(2)
            ],
        )


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
        everest_config_with_defaults(model={"realizations": realizations})


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
    config = everest_config_with_defaults(model={"realizations": realizations})
    assert config.model.realizations == expected


def test_that_nested_extra_types_are_validated_correctly(change_to_tmpdir):
    Path("everest_config.yml").write_text(
        dedent("""
        objective_functions:
          - name: my_func

        controls:
          - name: my_control
            type: generic_control
            min: 0
            max: 1
            initial_guess: 0.5
            perturbation_magnitude: 0.01
            variables:
                - name: my_var

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
    assert err.value.errors[0]["ctx"] == {"line_number": 18}
    assert err.value.errors[0]["type"] == "extra_forbidden"


@given(illegal_name=st.sampled_from(_RESERVED_WORDS))
def test_that_reserved_words_are_rejected(illegal_name):
    with pytest.raises(
        ValidationError,
        match=f"'{illegal_name}' is a reserved word and cannot be used.",
    ):
        everest_config_with_defaults(controls=[{"name": illegal_name}])

    with pytest.raises(
        ValidationError,
        match=f"'{illegal_name}' is a reserved word and cannot be used.",
    ):
        everest_config_with_defaults(objective_functions=[{"name": illegal_name}])


def test_forward_model_step_config_missing_type():
    expected_substring = (
        "Missing required field 'type' in 'results'. This field is needed to "
        "determine the correct result schema (e.g., 'gen_data' or 'summary')."
        " Please include a 'type' key in the 'results' section."
    )

    with pytest.raises(ValidationError) as exc_info:
        ForwardModelStepConfig(job="example_job", results={"file_name": "output.txt"})
    assert expected_substring in str(exc_info.value)


def test_ambiguous_max_memory_vs_realization_memory_is_detected():
    with pytest.raises(
        ValidationError, match="Ambiguous configuration of realization_memory"
    ):
        everest_config_with_defaults(
            simulator={
                "max_memory": "20",
                "queue_system": {"name": "local", "realization_memory": "40"},
            }
        )


@pytest.mark.parametrize(
    "max_memory, realization_memory, expected",
    [
        (None, 0, 0),
        (0, 0, 0),
        (55, 0, 55),
        (55, 55, 55),
    ],
)
def test_that_max_memory_propagates_to_realization_memory(
    max_memory, realization_memory, expected
) -> None:
    """Also testing that 0 for realization_memory means not set"""
    config = everest_config_with_defaults(
        simulator={
            "max_memory": max_memory,
            "queue_system": {"name": "local", "realization_memory": realization_memory},
        }
    )
    assert config.simulator.queue_system.realization_memory == expected


@pytest.mark.parametrize(
    "realization_memory, expected",
    [
        ("1Gb", 1073741824),
        ("2Kb", 2048),
        (999, 999),
    ],
)
def test_parsing_of_realization_memory(realization_memory, expected) -> None:
    config = everest_config_with_defaults(
        simulator={
            "queue_system": {"name": "local", "realization_memory": realization_memory},
        }
    )
    assert config.simulator.queue_system.realization_memory == expected


@pytest.mark.parametrize(
    "invalid_memory_spec, error_message",
    [
        ("-1", "Negative memory does not make sense"),
        ("      -2", "Negative memory does not make sense"),
        ("-1b", "Negative memory does not make sense in -1b"),
        ("b", "Invalid memory string"),
        ("'kljh3 k34f15gg.  asd '", "Invalid memory string"),
        ("'kljh3 1gb'", "Invalid memory string"),
        ("' 2gb 3k 1gb'", "Invalid memory string"),
        ("4ub", "Unknown memory unit"),
        ("1x", "Unknown memory unit"),
        ("1 x", "Unknown memory unit"),
        ("1 xy", "Unknown memory unit"),
        ("foo", "Invalid memory string: foo"),
    ],
)
def test_parsing_of_invalid_memory_spec(invalid_memory_spec, error_message) -> None:
    with pytest.raises(ValidationError, match=error_message):
        everest_config_with_defaults(
            simulator={
                "queue_system": {
                    "name": "local",
                    "realization_memory": invalid_memory_spec,
                },
            }
        )
    with pytest.raises(ValidationError, match=error_message):
        everest_config_with_defaults(simulator={"max_memory": invalid_memory_spec})


def test_parsing_of_unset_realization_memory() -> None:
    config = everest_config_with_defaults(
        simulator={
            "queue_system": {"name": "local"},
        }
    )
    assert config.simulator.queue_system.realization_memory == 0


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "max_memory",
    [
        None,
        0,
        1,
        "0",
        "1",
        "1b",
        "1k",
        "1m",
        "1g",
        "1t",
        "1p",
        "1G",
        "1 G",
        "1Gb",
        "1 Gb",
    ],
)
def test_that_max_memory_is_valid(max_memory) -> None:
    everest_config_with_defaults(simulator={"max_memory": max_memory})


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "max_memory",
    [-1, "-1", "-1G", "-1 G", "-1Gb"],
)
def test_that_negative_max_memory_fails(max_memory) -> None:
    with pytest.raises(
        ValidationError, match=f"Negative memory does not make sense in {max_memory}"
    ):
        everest_config_with_defaults(simulator={"max_memory": max_memory})
