import json
import uuid
from unittest.mock import MagicMock

import numpy as np

from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisMatrixEvent,
    DataSection,
)
from ert.run_models.update_run_model import UpdateRunModel


def test_that_send_smoother_event_persists_observation_report_on_analysis_complete():
    model = MagicMock(spec=UpdateRunModel)
    model._storage = MagicMock()
    mock_ensemble = MagicMock()
    model._storage.get_ensemble.return_value = mock_ensemble

    posterior_id = str(uuid.uuid4())
    data_section = DataSection(
        header=["observation_key", "status"],
        data=[("OBS_1", "Active"), ("OBS_2", "Deactivated, outlier")],
    )
    event = AnalysisCompleteEvent(data=data_section, posterior_id=posterior_id)

    UpdateRunModel.send_smoother_event(
        model, iteration=0, run_id=uuid.uuid4(), event=event
    )

    model._storage.get_ensemble.assert_called_once_with(posterior_id)
    mock_ensemble.save_transition_data.assert_called_once()

    _, saved_json = mock_ensemble.save_transition_data.call_args[0]
    parsed = json.loads(saved_json)
    assert parsed["posterior_id"] == posterior_id
    assert parsed["data"]["header"] == ["observation_key", "status"]
    assert parsed["data"]["data"] == [
        ["OBS_1", "Active"],
        ["OBS_2", "Deactivated, outlier"],
    ]


def test_that_send_smoother_event_delegates_matrix_to_save_transition_matrix():
    model = MagicMock(spec=UpdateRunModel)
    model._storage = MagicMock()
    mock_ensemble = MagicMock()
    mock_ensemble.save_transition_matrix.return_value = (
        "/some/path/corr_XY_PARAM.npy",
        False,
    )
    model._storage.get_ensemble.return_value = mock_ensemble

    posterior_id = str(uuid.uuid4())
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    event = AnalysisMatrixEvent(
        name="corr_XY_PARAM",
        posterior_id=posterior_id,
        matrix=matrix,
    )

    UpdateRunModel.send_smoother_event(
        model, iteration=0, run_id=uuid.uuid4(), event=event
    )

    model._storage.get_ensemble.assert_called_once_with(posterior_id)
    mock_ensemble.save_transition_matrix.assert_called_once()
    call_args = mock_ensemble.save_transition_matrix.call_args[0]
    assert call_args[0] == "corr_XY_PARAM"
    np.testing.assert_array_equal(call_args[1], matrix)

    mock_ensemble.save_transition_data.assert_called_once()
    file_name, saved_json = mock_ensemble.save_transition_data.call_args[0]
    assert file_name == "corr_XY_PARAM.json"
    parsed = json.loads(saved_json)
    assert parsed["event_type"] == "AnalysisStorageEvent"
    assert parsed["uri"] == "/some/path/corr_XY_PARAM.npy"
    assert parsed["ensemble_id"] == posterior_id
    assert parsed["sparse"] is False
