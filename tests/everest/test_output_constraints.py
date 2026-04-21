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

    initial_everest_config.write_to_file(str(config_file_path))

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


def test_that_auto_scale_and_constraints_scale_are_mutually_exclusive():
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
    initial_config.write_to_file(config_file_path)

    config = EverestConfig.load_file(str(config_file_path))

    ropt_conf, _ = everest2ropt(
        [ctrl for c in config.controls for ctrl in c.to_ert_parameter_config()],
        config.create_ert_objectives_config(),
        config.input_constraints,
        config.create_ert_output_constraints_config(),
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
        None,
        None,
        None,
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


def test_that_duplicate_output_constraint_names_raise_error():
    with pytest.raises(ValueError, match="Output constraint names must be unique"):
        everest_config_with_defaults(
            output_constraints=[
                {"target": 0.3, "name": "a"},
                {"target": 0.3, "name": "a"},
            ],
        )


def test_that_target_or_bounds_are_provided():
    with pytest.raises(
        ValueError,
        match=(r"(?s).*Must provide target or lower_bound/upper_bound.*"),
    ):
        everest_config_with_defaults(
            input_constraints=[
                OutputConstraintConfig.model_validate(
                    {
                        "name": "some_name",
                    }
                )
            ],
        )


def test_that_target_and_bounds_are_mutually_exclusive():
    with pytest.raises(
        ValueError,
        match=(r"(?s).*Cannot combine target and bounds.*"),
    ):
        everest_config_with_defaults(
            input_constraints=[
                OutputConstraintConfig.model_validate(
                    {"name": "some_name", "target": 1.0, "lower_bound": 0.0}
                )
            ],
        )


def test_that_lower_bound_cannot_be_greater_than_upper_bound():
    with pytest.raises(
        match=(r"(?s).*The upper_bound must be greater than the lower_bound.*"),
    ):
        everest_config_with_defaults(
            output_constraints=[
                OutputConstraintConfig.model_validate(
                    {
                        "name": "some_name",
                        "lower_bound": 2.0,
                        "upper_bound": 1.0,
                    }
                )
            ],
        )
