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
from ert.run_models import (
    EnsembleInformationFilter,
    EnsembleSmoother,
    ErtRunError,
    MultipleDataAssimilation,
)
from ert.run_models.manual_update import ManualUpdate
from ert.run_models.manual_update_enif import ManualUpdateEnIF
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Storage


@pytest.fixture(
    params=[
        EnsembleSmoother,
        EnsembleInformationFilter,
        MultipleDataAssimilation,
        ManualUpdate,
        ManualUpdateEnIF,
    ],
    ids=lambda model_type: model_type.__name__,
)
def update_model(
    request: pytest.FixtureRequest, storage: Storage
) -> tuple[type[UpdateRunModel], MagicMock]:
    model = MagicMock(spec=request.param)
    model._storage = MagicMock(wraps=storage)
    model._run_paths = MagicMock()
    model.update_settings = ObservationSettings()
    model.analysis_settings = ESSettings()
    model.random_seed = 123
    return request.param, model


@pytest.mark.parametrize("has_parameters", [False, True], ids=["empty", "all-disabled"])
def test_that_update_rejects_non_updatable_prior_before_creating_posterior(
    update_model: tuple[type[UpdateRunModel], MagicMock],
    storage: Storage,
    has_parameters: bool,
) -> None:
    model_type, model = update_model
    parameter = GenKwConfig(
        name="PARAMETER", distribution={"name": "normal", "mean": 0, "std": 1}
    )
    model.parameter_configuration = [parameter]
    fixed_parameter = parameter.model_copy(update={"update_strategy": None})
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": (
                [fixed_parameter.model_dump(mode="json")] if has_parameters else []
            )
        }
    )
    prior = storage.create_ensemble(experiment, name="prior", ensemble_size=2)

    with pytest.raises(ErtRunError, match="No parameters to update") as exc_info:
        model_type.update(model, prior, "posterior")

    assert f"Cannot update prior ensemble '{prior.name}' (ID: {prior.id})" in str(
        exc_info.value
    )
    assert isinstance(exc_info.value.__cause__, ConfigValidationError)
    model._storage.create_ensemble.assert_not_called()
    model.run_workflows.assert_not_called()
    model.update_ensemble_parameters.assert_not_called()
    model.send_event.assert_not_called()


@pytest.mark.parametrize("has_parameters", [False, True], ids=["empty", "all-disabled"])
def test_that_update_uses_updatable_prior_despite_non_updatable_current_config(
    update_model: tuple[type[UpdateRunModel], MagicMock],
    storage: Storage,
    has_parameters: bool,
) -> None:
    model_type, model = update_model
    parameter = GenKwConfig(
        name="PARAMETER", distribution={"name": "normal", "mean": 0, "std": 1}
    )
    fixed_parameter = parameter.model_copy(
        update={"name": "FIXED", "update_strategy": None}
    )
    model.parameter_configuration = [fixed_parameter] if has_parameters else []
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [
                fixed_parameter.model_dump(mode="json"),
                parameter.model_dump(mode="json"),
            ]
        }
    )
    prior = storage.create_ensemble(experiment, name="prior", ensemble_size=2)

    posterior = model_type.update(model, prior, "posterior", weight=2.0)

    assert posterior.name == "posterior"
    assert posterior.iteration == prior.iteration + 1
    model._storage.create_ensemble.assert_called_once()
    model.update_ensemble_parameters.assert_called_once_with(prior, posterior, 2.0)


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
