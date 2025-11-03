import numpy as np
import pytest

from ert.base_model_context import use_runtime_plugins
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, OptimizationConfig, OutputConstraintConfig
from everest.optimizer.everest2ropt import everest2ropt
from tests.everest.utils import everest_config_with_defaults


def test_constraints_init(tmp_path):
    num_constraints = 16
    initial_everest_config = everest_config_with_defaults(
        output_constraints=[
            OutputConstraintConfig(
                name=f"oil_prod_rate_{i:03d}", upper_bound=5000.0, scale=7500.0
            )
            for i in range(num_constraints)
        ]
    )

    config_file_path = tmp_path / "mocked_output_constraints.yml"

    initial_everest_config.dump(str(config_file_path))

    loaded_everest_config = EverestConfig.load_file(str(config_file_path))

    constraints_from_loaded_config = loaded_everest_config.output_constraints

    constraint_names = [cn.name for cn in constraints_from_loaded_config]
    assert constraint_names == [
        f"oil_prod_rate_{i:03d}" for i in range(num_constraints)
    ]

    assert [
        cn.upper_bound for cn in constraints_from_loaded_config
    ] == num_constraints * [5000.0]
    assert [cn.scale for cn in constraints_from_loaded_config] == num_constraints * [
        7500.0
    ]


@pytest.mark.parametrize(
    "config, error",
    [
        (
            [
                {"name": "some_name"},
            ],
            "Must provide target or lower_bound/upper_bound",
        ),
        (
            [
                {"name": "same_name", "upper_bound": 5000},
                {"name": "same_name", "upper_bound": 5000},
            ],
            "Output constraint names must be unique",
        ),
        (
            [
                {"name": "some_name", "upper_bound": 5000, "target": 5000},
            ],
            r"Can not combine target and bounds",
        ),
        (
            [
                {"name": "some_name", "lower_bound": 10, "upper_bund": 2},
            ],
            "Extra inputs are not permitted",
        ),
        (
            [
                {"name": "some_name", "upper_bound": "2ooo"},
            ],
            "unable to parse string as a number",
        ),
    ],
)
def test_output_constraint_config(config, error):
    with pytest.raises(ValueError, match=error):
        EverestConfig(
            output_constraints=config,
        )


def test_that_auto_scale_and_constraints_scale_are_mutually_exclusive(tmp_path):
    with pytest.raises(
        ValueError,
        match=(
            "The auto_scale option in the optimization section and the scale "
            "options in the output_constraints section are mutually exclusive"
        ),
    ):
        everest_config_with_defaults(
            optimization=OptimizationConfig(auto_scale=True),
            output_constraints=[
                OutputConstraintConfig(
                    name=f"oil_prod_rate_{i:03d}", upper_bound=5000.0, scale=7500.0
                )
                for i in range(2)
            ],
        )


def test_upper_bound_output_constraint_def(tmp_path):
    output_constraint_definition = OutputConstraintConfig(
        name="some_name",
        upper_bound=5000.0,
        scale=1.0,
    )

    initial_config = everest_config_with_defaults(
        output_constraints=[output_constraint_definition],
    )

    config_file_path = tmp_path / "test_config.yml"
    initial_config.dump(str(config_file_path))

    config = EverestConfig.load_file(str(config_file_path))

    ropt_conf, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )

    expected_nonlinear_constraint_representation = {
        "name": "some_name",
        "lower_bounds": -np.inf,
        "upper_bounds": [5000.0],
    }

    assert (
        expected_nonlinear_constraint_representation["lower_bounds"]
        == ropt_conf["nonlinear_constraints"]["lower_bounds"][0]
    )
    assert (
        expected_nonlinear_constraint_representation["upper_bounds"]
        == ropt_conf["nonlinear_constraints"]["upper_bounds"]
    )

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        EverestRunModel.create(config, runtime_plugins=site_plugins)
