import pytest

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt


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
    ],
)
def test_config_multi_objectives(min_config, config, expectation, tmp_path):
    min_config["objective_functions"] = config

    with expectation:
        EverestConfig(**min_config)


def test_multi_objectives2ropt():
    weights = [1.33, 3.1]
    config = EverestConfig.with_defaults(
        objective_functions=[
            {"name": "f1", "weight": weights[0]},
            {"name": "f2", "weight": weights[1]},
        ]
    )
    enopt_config, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )
    assert len(enopt_config["objectives"]["weights"]) == 2
    assert enopt_config["objectives"]["weights"][0] == weights[0]
    assert enopt_config["objectives"]["weights"][1] == weights[1]
