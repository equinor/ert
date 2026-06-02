import io
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from ert.analysis.event import AnalysisCompleteEvent, DataSection
from ert.run_models.update_run_model import UpdateRunModel


def test_that_send_smoother_event_persists_observation_report_on_analysis_complete():
    model = MagicMock(spec=UpdateRunModel)
    model._storage = MagicMock()
    mock_ensemble = MagicMock()
    ensemble_id = str(uuid.uuid4())
    mock_ensemble.id = uuid.UUID(ensemble_id)
    model._storage.get_ensemble.return_value = mock_ensemble

    data_section = DataSection(
        header=["observation_key", "status"],
        data=[("OBS_1", "Active"), ("OBS_2", "Deactivated, outlier")],
    )
    event = AnalysisCompleteEvent(data=data_section, ensemble_id=ensemble_id)

    UpdateRunModel.send_smoother_event(
        model, iteration=0, run_id=uuid.uuid4(), event=event
    )

    model._storage.get_ensemble.assert_called_once_with(ensemble_id)
    mock_ensemble.save_blob.assert_called_once()

    saved_bytes = mock_ensemble.save_blob.call_args[0][0]
    assert isinstance(saved_bytes, bytes)
    saved_df = pl.read_parquet(io.BytesIO(saved_bytes))
    assert saved_df.columns == ["observation_key", "status"]
    assert len(saved_df) == 2
    assert saved_df["observation_key"].to_list() == ["OBS_1", "OBS_2"]
    assert saved_df["status"].to_list() == ["Active", "Deactivated, outlier"]


def test_update_ensemble_parameters_uses_posterior_experiment_for_strategy_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    model = MagicMock()
    model.analysis_settings = SimpleNamespace(
        enkf_truncation=0.99,
        distance_localization=True,
        localization=False,
        correlation_threshold=None,
    )
    model.update_settings = MagicMock()
    model._rng = MagicMock()
    model.active_realizations = None

    prior = MagicMock()
    prior.iteration = 0
    prior.id = uuid.uuid4()
    prior.experiment.update_parameters = ["PARAM"]
    prior.experiment.parameter_configuration = {"PARAM": MagicMock()}
    prior.experiment.observation_keys = ["OBS"]

    posterior = MagicMock()
    posterior.experiment = MagicMock()

    captured: dict[str, object] = {}

    def build_strategy_map_spy(**kwargs):
        captured.update(kwargs)
        return {"PARAM": MagicMock()}

    monkeypatch.setattr(
        "ert.run_models.update_run_model.build_strategy_map",
        build_strategy_map_spy,
    )
    smoother_update = MagicMock()
    monkeypatch.setattr(
        "ert.run_models.update_run_model.smoother_update",
        smoother_update,
    )

    UpdateRunModel.update_ensemble_parameters(model, prior, posterior, weight=1.0)

    assert captured["experiment"] is posterior.experiment
    smoother_update.assert_called_once()
