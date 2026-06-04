import io
import uuid
from unittest.mock import MagicMock

import polars as pl

from ert.analysis.event import (
    AnalysisCompleteEvent,
    DataSection,
)
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
    event = AnalysisCompleteEvent(
        update_algorithm="ensemble_smoother",
        data=data_section,
    )

    UpdateRunModel.send_smoother_event(
        model, iteration=0, run_id=uuid.uuid4(), ensemble=mock_ensemble, event=event
    )

    mock_ensemble.save_blob.assert_called_once()

    saved_bytes = mock_ensemble.save_blob.call_args[0][0]
    assert isinstance(saved_bytes, bytes)
    saved_df = pl.read_parquet(io.BytesIO(saved_bytes))
    assert saved_df.columns == ["observation_key", "status"]
    assert len(saved_df) == 2
    assert saved_df["observation_key"].to_list() == ["OBS_1", "OBS_2"]
    assert saved_df["status"].to_list() == ["Active", "Deactivated, outlier"]
