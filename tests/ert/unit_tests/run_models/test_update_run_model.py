import uuid
from unittest.mock import MagicMock

import pytest

from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
    AnalysisErrorEvent,
    AnalysisStatusEvent,
    DataSection,
)
from ert.run_models.update_run_model import UpdateRunModel

_DATA_SECTION = DataSection(
    header=["observation_key", "status"],
    data=[("OBS_1", "Active"), ("OBS_2", "Deactivated, outlier")],
)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            AnalysisCompleteEvent(
                data=_DATA_SECTION, update_algorithm="ensemble_smoother"
            ),
            id="complete",
        ),
        pytest.param(
            AnalysisDataEvent(name="Auto scale: OBS_GROUP", data=_DATA_SECTION),
            id="data",
        ),
        pytest.param(
            AnalysisErrorEvent(
                error_msg="No active observations left",
                data=_DATA_SECTION,
                update_algorithm="ensemble_smoother",
            ),
            id="error",
        ),
    ],
)
def test_that_send_smoother_event_persists_update_tables_on_posterior_ensemble(event):
    model = MagicMock(spec=UpdateRunModel)
    mock_ensemble = MagicMock()

    UpdateRunModel.send_smoother_event(
        model,
        iteration=0,
        run_id=uuid.uuid4(),
        ensemble=mock_ensemble,
        event=event,
    )

    mock_ensemble.save_blob.assert_called_once_with(event)


def test_that_send_smoother_event_does_not_persist_status_messages():
    model = MagicMock(spec=UpdateRunModel)
    mock_ensemble = MagicMock()

    UpdateRunModel.send_smoother_event(
        model,
        iteration=0,
        run_id=uuid.uuid4(),
        ensemble=mock_ensemble,
        event=AnalysisStatusEvent(msg="Loading data"),
    )

    mock_ensemble.save_blob.assert_not_called()
