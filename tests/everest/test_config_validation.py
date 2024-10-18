import os
import pathlib
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest
from pydantic import ValidationError

from everest.config import EverestConfig, ModelConfig
from everest.config.control_variable_config import ControlVariableConfig
from everest.config.sampler_config import SamplerConfig


def has_error(error: Union[ValidationError, List[dict]], match: str):
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
        EverestConfig.with_defaults(**{"optimization": {"cvar": cvar}})

    assert has_error(e.value, match="Invalid CVaR section")


@pytest.mark.parametrize("nreals", [-1, 0, 8])
def test_that_cvar_nreals_interval_outside_range_errors(nreals):
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            **{
                "optimization": {
                    "cvar": {
                        "number_of_realizations": nreals,
                    }
                },
                "model": {"realizations": [1, 2, 3, 4, 5, 6]},
            }
        )

    assert has_error(
        e.value,
        match=f"number_of_realizations: \\(got {nreals}",
    )


@pytest.mark.parametrize("nreals", [1, 2, 3, 4, 5])
def test_that_cvar_nreals_valid_doesnt_error(nreals):
    EverestConfig.with_defaults(
        **{
            "optimization": {
                "cvar": {
                    "number_of_realizations": nreals,
                }
            },
            "model": {"realizations": [1, 2, 3, 4, 5, 6]},
        }
    )


def test_that_max_runtime_errors_only_on_negative():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(simulator={"max_runtime": -1})

    EverestConfig.with_defaults(simulator={"max_runtime": 0})

    assert has_error(e.value, match=".*greater than or equal to 0")


def test_that_invalid_queue_system_errors():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(simulator={"queue_system": "docal"})

    assert has_error(
        e.value, match="Input should be 'lsf', 'local', 'slurm' or 'torque'"
    )
    EverestConfig.with_defaults(simulator={"queue_system": "local"})
    EverestConfig.with_defaults(simulator={"queue_system": "lsf"})
    EverestConfig.with_defaults(simulator={"queue_system": "slurm"})
    EverestConfig.with_defaults(simulator={"queue_system": "torque"})


@pytest.mark.parametrize(
    ["cores", "expected_error"], [(0, True), (-1, True), (1, False)]
)
def test_that_cores_errors_only_on_lt0(cores, expected_error):
    if expected_error:
        with pytest.raises(ValueError) as e:
            EverestConfig.with_defaults(simulator={"cores": cores})

        assert has_error(e.value, match=".*greater than 0")

        with pytest.raises(ValueError) as e:
            EverestConfig.with_defaults(simulator={"cores_per_node": cores})

        assert has_error(e.value, match=".*greater than 0")
    else:
        EverestConfig.with_defaults(simulator={"cores": cores})
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
    variables: List[Dict[str, Any]], count: int
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
        EverestConfig.with_defaults(**{"model": {"realizations": [-1, 1, 2, 3]}})

    assert has_error(e.value, match="Input should be greater than or equal to 0")

    EverestConfig.with_defaults(**{"model": {"realizations": [0, 1, 2, 3]}})


def test_that_model_realizations_weights_must_correspond_to_realizations():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            **{"model": {"realizations": [1, 2, 3], "realizations_weights": [1, 2]}}
        )
    assert has_error(
        e.value, match="Specified realizations_weights must have one weight per"
    )

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            **{
                "model": {
                    "realizations": [1, 2, 3],
                    "realizations_weights": [1, 2, 3, 4],
                }
            }
        )
    assert has_error(
        e.value, match="Specified realizations_weights must have one weight per"
    )

    EverestConfig.with_defaults(**{"model": {"realizations": [1, 2, 3]}})
    EverestConfig.with_defaults(
        **{"model": {"realizations": [1, 2, 3], "realizations_weights": [5, 5, -5]}}
    )


def test_that_missing_optimization_algorithm_errors():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(**{"optimization": {"algorithm": "ddlygldt"}})

    assert has_error(e.value, match="Optimizer algorithm 'dakota/ddlygldt' not found")


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
        EverestConfig.with_defaults(**{"optimization": {optimizer_attr: -1}})

    assert has_error(e.value, match="(.*)Input should be greater than 0")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(**{"optimization": {optimizer_attr: 0}})

    assert has_error(e.value, match="(.*)Input should be greater than 0")

    EverestConfig.with_defaults(**{"optimization": {optimizer_attr: 1}})


def test_that_min_realizations_success_is_nonnegative():
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            **{"optimization": {"min_realizations_success": -1}}
        )

    assert has_error(e.value, match="(.*)Input should be greater than or equal to 0")

    EverestConfig.with_defaults(**{"optimization": {"min_realizations_success": 0}})


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


def test_that_model_report_steps_invalid_dates_errors(change_to_tmpdir):
    os.makedirs("config_dir/relative/path")
    with open("config_dir/test.yml", "w", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
                "report_steps": ["2022-02-02", "hey", "yo", "sup", "ma", "dawg"],
                "data_file": "relative/path",
            },
            config_path=Path("config_dir/test.yml"),
        )

    assert has_error(e.value, "malformed dates: hey, yo, sup, ma, dawg")

    EverestConfig.with_defaults(
        model={
            "realizations": [1, 2, 3],
            "report_steps": ["2022-01-01", "2022-01-03", "2022-01-05"],
            "data_file": "relative/path",
        },
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
    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
            model={
                "realizations": [1, 2, 3],
            },
            config_path=Path("config_dir/test.yml"),
            **{install_keyword: [{"name": "test", "source": "non_existing"}]},
        )

    assert has_error(e.value, "No such file or directory")


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

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
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

    assert has_error(e.value, "malformed EXECUTABLE in malformed.ert")
    assert has_error(e.value, "malformed EXECUTABLE in malformed2.ert")


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

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
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

    assert has_error(e.value, ".*non_executable is not executable")


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

    with pytest.raises(ValueError) as e:
        EverestConfig.with_defaults(
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

    assert has_error(e.value, "No such executable non_existing")


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
        EverestConfig(**{})

    error_dicts = e.value.errors()

    # Expect missing error for:
    # controls, objective_functions, config_path
    assert len(error_dicts) == 3

    config_with_defaults = EverestConfig.with_defaults()
    config_args = {}
    required_argnames = [
        "controls",
        "objective_functions",
        "config_path",
    ]

    for key in required_argnames:
        with pytest.raises(ValidationError) as e:
            EverestConfig(**config_args)

        assert len(e.value.errors()) == len(required_argnames) - len(config_args)
        config_args[key] = getattr(config_with_defaults, key)


def test_that_non_existing_workflow_jobs_cause_error():
    with pytest.raises(ValidationError, match="No such file or directory (.*)jobs/JOB"):
        EverestConfig.with_defaults(
            install_workflow_jobs=[{"name": "job0", "source": "jobs/JOB"}],
            workflows={
                "pre_simulation": [
                    "job0 -i in -o out",
                    "job1 -i out -o result",
                ]
            },
        )
