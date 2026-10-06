import uuid
from unittest.mock import MagicMock

import pytest

from ert.analysis.event import AnalysisCompleteEvent, DataSection
from ert.config import (
    ConfigValidationError,
    ESSettings,
    GenKwConfig,
    ObservationSettings,
)
from ert.run_models import ErtRunError
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Storage


@pytest.fixture
def update_model(storage: Storage) -> MagicMock:
    model = MagicMock(spec=UpdateRunModel)
    model._validate_has_updatable_parameter = (
        UpdateRunModel._validate_has_updatable_parameter
    )
    model._storage = MagicMock(wraps=storage)
    model._runpaths = MagicMock()
    model.update_settings = ObservationSettings()
    model.analysis_settings = ESSettings()
    model.random_seed = 123
    return model


def test_that_update_rejects_non_updatable_prior_before_creating_posterior(
    update_model: MagicMock, storage: Storage
):
    parameter = GenKwConfig(
        name="PARAMETER",
        distribution={"name": "normal", "mean": 0, "std": 1},
        update_strategy=None,
    )
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [parameter.model_dump(mode="json")]
        }
    )
    prior = storage.create_ensemble(experiment, name="prior", ensemble_size=2)

    with pytest.raises(ErtRunError, match="No parameters to update") as exc_info:
        UpdateRunModel.update(update_model, prior, "posterior")

    assert f"Cannot update prior ensemble '{prior.name}' (ID: {prior.id})" in str(
        exc_info.value
    )
    assert isinstance(exc_info.value.__cause__, ConfigValidationError)
    update_model._storage.create_ensemble.assert_not_called()
    update_model.run_workflows.assert_not_called()
    update_model.update_ensemble_parameters.assert_not_called()


def test_that_update_creates_next_iteration_from_updatable_prior(
    update_model: MagicMock, storage: Storage
):
    parameter = GenKwConfig(
        name="PARAMETER", distribution={"name": "normal", "mean": 0, "std": 1}
    )
    fixed_parameter = parameter.model_copy(
        update={"name": "FIXED", "update_strategy": None}
    )

    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [
                fixed_parameter.model_dump(mode="json"),
                parameter.model_dump(mode="json"),
            ]
        }
    )
    updatable_prior = storage.create_ensemble(experiment, name="prior", ensemble_size=2)

    posterior = UpdateRunModel.update(
        update_model, updatable_prior, "posterior", weight=2.0
    )

    assert posterior.name == "posterior"
    assert posterior.iteration == updatable_prior.iteration + 1
    update_model._storage.create_ensemble.assert_called_once()
    update_model.update_ensemble_parameters.assert_called_once_with(
        updatable_prior, posterior, 2.0
    )


def test_that_send_smoother_event_persists_observation_report_on_analysis_complete():
    model = MagicMock(spec=UpdateRunModel)
    mock_ensemble = MagicMock()

    data_section = DataSection(
        header=["observation_key", "status"],
        data=[("OBS_1", "Active"), ("OBS_2", "Deactivated, outlier")],
    )
    event = AnalysisCompleteEvent(
        data=data_section, update_algorithm="ensemble_smoother"
    )

    UpdateRunModel.send_smoother_event(
        model,
        iteration=0,
        run_id=uuid.uuid4(),
        ensemble=mock_ensemble,
        event=event,
    )

    mock_ensemble.save_blob.assert_called_once_with(event)
