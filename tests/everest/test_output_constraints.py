import numpy as np
import pytest

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt

CONFIG_FILE = "config_output_constraints.yml"


def test_constraints_init(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    constr = config.output_constraints

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


def test_upper_bound_output_constraint_def(copy_mocked_test_data_to_tmp):
    with open("conf_file", "w", encoding="utf-8") as f:
        f.write(" ")

    config = EverestConfig.with_defaults(
        output_constraints=[
            {"name": "some_name", "upper_bound": 5000, "scale": 1.0},
        ],
    )

    # Check ropt conversion
    ropt_conf, _ = everest2ropt(config)

    expected = {
        "name": "some_name",
        "lower_bounds": -np.inf,
        "upper_bounds": [5000],
    }

    assert (
        expected["lower_bounds"]
        == ropt_conf["nonlinear_constraints"]["lower_bounds"][0]
    )
    assert (
        expected["upper_bounds"] == ropt_conf["nonlinear_constraints"]["upper_bounds"]
    )

    EverestRunModel.create(config)


@pytest.mark.integration_test
def test_sim_output_constraints(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
