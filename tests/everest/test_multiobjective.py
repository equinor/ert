from contextlib import ExitStack as does_not_raise

import pytest
from ropt.config.enopt import EnOptConfig

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt

CONFIG_FILE = "config_multi_objectives.yml"


@pytest.mark.parametrize(
    "config, expectation",
    [
        (
            [{"name": "c1", "weight": 1.0}, {"name": "c2"}],
            pytest.raises(
                ValueError,
                match="Weight should be given either for all of the"
                " objectives or for none of them",
            ),
        ),
        (
            [{"weight": 1.0}],
            pytest.raises(
                ValueError,
                match="Field required",
            ),
        ),
        (
            [{"name": "c1", "weight": -1.0}],
            pytest.raises(
                ValueError,
                match="The objective weight should be greater than 0",
            ),
        ),
        (
            [{"name": "c1", "weight": 0}],
            pytest.raises(
                ValueError,
                match="The objective weight should be greater than 0",
            ),
        ),
        (
            [{"name": "c1", "scale": 0}],
            pytest.raises(
                ValueError,
                match="Scale value cannot be zero",
            ),
        ),
        (
            [{"name": "c1", "scale": -125}],
            does_not_raise(),
        ),
    ],
)
def test_config_multi_objectives(min_config, config, expectation, tmp_path):
    min_config["objective_functions"] = config

    with expectation:
        EverestConfig(**min_config)


def test_multi_objectives2ropt(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    config_dict = config.to_dict()
    ever_objs = config_dict["objective_functions"]
    ever_objs[0]["weight"] = 1.33
    ever_objs[1]["weight"] = 3.1
    assert len(EverestConfig.lint_config_dict(config_dict)) == 0

    norm = ever_objs[0]["weight"] + ever_objs[1]["weight"]

    enopt_config = EnOptConfig.model_validate(
        everest2ropt(EverestConfig.model_validate(config_dict))
    )
    assert len(enopt_config.objectives.weights) == 2
    assert enopt_config.objectives.weights[1] == ever_objs[1]["weight"] / norm
    assert enopt_config.objectives.weights[0] == ever_objs[0]["weight"] / norm


@pytest.mark.integration_test
def test_multi_objectives_run(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
