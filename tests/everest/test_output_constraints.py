import numpy as np
import pytest
from ropt.config.enopt import EnOptConfig

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt

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
    with pytest.raises(
        ValueError, match="Output constraints must have only one of the following"
    ):
        EverestConfig(
            output_constraints=[
                {"name": "some_name"},
            ],
        )

    # Same name
    with pytest.raises(ValueError, match="Output constraint names must be unique"):
        EverestConfig(
            output_constraints=[
                {"name": "same_name", "upper_bound": 5000},
                {"name": "same_name", "upper_bound": 5000},
            ],
        )

    # Two RHS
    with pytest.raises(
        ValueError,
        match=r"Output constraints must have only one of the following:"
        " { target }, or { upper and/or lower bound }",
    ):
        EverestConfig(
            output_constraints=[
                {"name": "some_name", "upper_bound": 5000, "target": 5000},
            ],
            model={"realizations": [0]},
        )

    # Wrong RHS attribute
    wrong_rhs_config = {
        "output_constraints": [
            {"name": "some_name", "target": 2},
        ],
    }

    wrong_rhs_config["output_constraints"][0]["upper_bund"] = 5000

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EverestConfig(**wrong_rhs_config)

    # Wrong RHS type
    with pytest.raises(ValueError, match="unable to parse string as a number"):
        EverestConfig(
            output_constraints=[
                {"name": "some_name", "upper_bound": "2ooo"},
            ],
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
    ropt_conf = EnOptConfig.model_validate(everest2ropt(config))

    expected = {
        "name": "some_name",
        "lower_bounds": -np.inf,
        "upper_bounds": [5000],
    }

    assert expected["lower_bounds"] == ropt_conf.nonlinear_constraints.lower_bounds[0]
    assert expected["upper_bounds"] == ropt_conf.nonlinear_constraints.upper_bounds[0]

    EverestRunModel.create(config)


@pytest.mark.integration_test
def test_sim_output_constraints(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
