import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ert.analysis import build_strategy_map
from ert.config import ESSettings, GenKwConfig, LocalizationType
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import MultipleDataAssimilation as mda
from ert.run_models.manual_update import ManualUpdate
from ert.run_models.multiple_data_assimilation import MultipleDataAssimilation
from ert.run_models.run_model import ErtRunError
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Storage, open_storage


@pytest.mark.parametrize("model_type", [MultipleDataAssimilation, ManualUpdate])
@pytest.mark.parametrize("prior_iteration", [0, 1])
def test_that_prior_overrides_apply_to_each_update_and_only_persist_in_the_target(
    storage: Storage, model_type, prior_iteration
):
    parameter = GenKwConfig(
        name="stored", distribution={"name": "uniform", "min": 0, "max": 1}
    )
    original = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [parameter.model_dump(mode="json")]
        }
    )
    prior = original.create_ensemble(
        name="prior", ensemble_size=3, iteration=prior_iteration
    )
    model = MagicMock(spec=model_type)
    model.parameter_update_overrides = {"stored": LocalizationType.ADAPTIVE}
    model.to_experiment_config.return_value = {}
    model.analysis_settings = ESSettings()
    model.update_settings = MagicMock()
    model.active_realizations = [True] * 3
    model._rng = np.random.default_rng(0)
    model._prior = prior
    model._storage = storage
    model.experiment_name = "edited"

    if model_type is MultipleDataAssimilation:
        target_config = MultipleDataAssimilation._create_experiment_from_prior(
            model, original
        )
        target = storage.create_experiment(experiment_config=target_config)
    else:
        target = ManualUpdate._create_experiment_storage(model)
    posterior = target.create_ensemble(
        name="posterior",
        ensemble_size=3,
        iteration=prior.iteration + 1,
        prior_ensemble=prior,
    )
    with (
        patch(
            "ert.run_models.update_run_model.build_strategy_map",
            wraps=build_strategy_map,
        ) as builder,
        patch("ert.run_models.update_run_model.smoother_update") as update,
    ):
        for source in [prior, posterior]:
            UpdateRunModel.update_ensemble_parameters(model, source, posterior, 1.0)
            assert (
                builder.call_args.kwargs["param_configs"]["stored"].update_strategy
                == LocalizationType.ADAPTIVE
            )
            assert builder.call_args.kwargs["experiment"] is source.experiment
            assert "stored" in update.call_args.kwargs["strategy_map"]

    assert (
        original.parameter_configuration["stored"].update_strategy
        == LocalizationType.GLOBAL
    )
    path, original_id, target_id = storage.path, original.id, target.id
    storage.close()
    with open_storage(path) as reopened:
        assert (
            reopened.get_experiment(original_id)
            .parameter_configuration["stored"]
            .update_strategy
            == LocalizationType.GLOBAL
        )
        assert (
            reopened.get_experiment(target_id)
            .parameter_configuration["stored"]
            .update_strategy
            == LocalizationType.ADAPTIVE
        )


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("2, 2, 2, 2", [4] * 4),
        ("1, 2, 4, ", [1.75, 3.5, 7.0]),
        ("1.414213562373095, 1.414213562373095", [2, 2]),
    ],
)
def test_that_parse_weights_returns_expected_values(weights, expected):
    weights = mda.parse_weights(weights)
    assert weights == expected
    assert math.isclose(np.reciprocal(weights).sum(), 1.0)


def test_that_non_numeric_weight_raises_value_error():
    with pytest.raises(ValueError, match="could not convert string to float: 'error'"):
        mda.parse_weights("2, error, 2, 2")


@pytest.mark.parametrize(
    "weights",
    [
        "2, -1, 2, 2",
        "2.0, 0.0, 2, 2",
        "-1, -1, -1, 0",
        "0, 0, 0, 0",
        "0.0,1.0, 2.0, 3.0",
    ],
)
def test_that_zero_or_negative_weights_raise_value_error(weights):
    with pytest.raises(
        ValueError,
        match=f"Invalid weights: {weights}. Weights must be positive non zero numbers.",
    ):
        mda.parse_weights(weights)


def test_that_mda_rejects_non_updatable_prior_before_creating_experiment(
    storage: Storage,
) -> None:
    parameter = GenKwConfig(
        name="PARAMETER",
        distribution={"name": "uniform", "min": 0.8, "max": 1.2},
        update_strategy=None,
    )
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [parameter.model_dump(mode="json")]
        }
    )
    prior = storage.create_ensemble(experiment, name="prior", ensemble_size=1)

    model = MagicMock(spec=MultipleDataAssimilation)
    model.prior_ensemble_id = str(prior.id)
    model._start_iteration = prior.iteration + 1
    model.analysis_settings = MagicMock()
    model.analysis_settings.weights = "1"
    model._storage = MagicMock()
    model._storage.get_ensemble.return_value = prior
    model._validate_has_updatable_parameter = (
        UpdateRunModel._validate_has_updatable_parameter
    )

    with pytest.raises(ErtRunError, match="No parameters to update"):
        MultipleDataAssimilation.run_experiment(
            model, MagicMock(spec=EvaluatorServerConfig)
        )

    model._storage.create_experiment.assert_not_called()
    model.update.assert_not_called()
