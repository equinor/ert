import uuid
from unittest.mock import MagicMock

from ert.analysis.event import AnalysisCompleteEvent, DataSection
from ert.run_models.update_run_model import UpdateRunModel


def test_that_send_smoother_event_persists_observation_report_on_analysis_complete():
    model = MagicMock(spec=UpdateRunModel)
    mock_ensemble = MagicMock()

    data_section = DataSection(
        header=["observation_key", "status"],
        data=[("OBS_1", "Active"), ("OBS_2", "Deactivated, outlier")],
    )
    event = AnalysisCompleteEvent(data=data_section)

    UpdateRunModel.send_smoother_event(
        model,
        iteration=0,
        run_id=uuid.uuid4(),
        ensemble=mock_ensemble,
        event=event,
    )

    mock_ensemble.save_blob.assert_called_once_with(event)
