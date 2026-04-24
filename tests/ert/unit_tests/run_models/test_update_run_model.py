import json
import uuid
from unittest.mock import MagicMock, PropertyMock

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


def test_that_send_smoother_event_saves_matrix_npy_on_analysis_matrix_event(tmp_path):
    model = MagicMock(spec=UpdateRunModel)
    model._storage = MagicMock()
    mock_ensemble = MagicMock()
    type(mock_ensemble).mount_point = PropertyMock(return_value=tmp_path)
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

    npy_path = tmp_path / "transition" / "corr_XY_PARAM.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path)
    np.testing.assert_array_equal(loaded, matrix)

    model._storage.get_ensemble.assert_called_once_with(posterior_id)
    mock_ensemble.save_transition_data.assert_called_once()

    _, saved_json = mock_ensemble.save_transition_data.call_args[0]
    parsed = json.loads(saved_json)
    assert parsed["name"] == "corr_XY_PARAM"
    assert parsed["posterior_id"] == posterior_id
    assert "matrix" not in parsed
