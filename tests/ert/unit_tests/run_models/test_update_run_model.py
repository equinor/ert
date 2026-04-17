import json
import uuid
from unittest.mock import MagicMock

from ert.analysis.event import AnalysisCompleteEvent, DataSection
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

    saved_name, saved_json = mock_ensemble.save_transition_data.call_args[0]
    assert saved_name == f"{AnalysisCompleteEvent.__name__}.json"
    parsed = json.loads(saved_json)
    assert parsed["posterior_id"] == posterior_id
    assert parsed["data"]["header"] == ["observation_key", "status"]
    assert parsed["data"]["data"] == [
        ["OBS_1", "Active"],
        ["OBS_2", "Deactivated, outlier"],
    ]
