import logging
import uuid
from unittest.mock import MagicMock

import polars as pl

from ert.analysis.event import AnalysisCompleteEvent, DataSection
from ert.analysis.snapshots import ObservationStatus, SmootherSnapshot
from ert.config.observation_quality_control import RFT_LOCATION_NOT_IN_GRID_ERROR
from ert.run_models.update_run_model import UpdateRunModel


def _smoother_snapshot(rows: list[dict[str, str]]) -> SmootherSnapshot:
    return SmootherSnapshot(
        source_ensemble_name="prior",
        target_ensemble_name="posterior",
        alpha=1.0,
        std_cutoff=1e-6,
        global_scaling=1.0,
        observations_and_responses=pl.DataFrame(
            rows,
            schema={"status": pl.String, "missing_realizations": pl.String},
        ),
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


def test_that_rft_grid_deactivation_log_reports_update_step_count_and_total(caplog):
    model = MagicMock(spec=UpdateRunModel)

    prior = MagicMock()
    prior.iteration = 2
    prior.experiment.observations = {
        "rft": pl.DataFrame({"observation_key": ["a", "b", "c", "d"]})
    }

    snapshot = _smoother_snapshot(
        [
            {"status": ObservationStatus.ACTIVE, "missing_realizations": ""},
            {"status": ObservationStatus.ACTIVE, "missing_realizations": ""},
            {
                "status": ObservationStatus.MISSING_RESPONSE,
                "missing_realizations": (
                    f"0: {RFT_LOCATION_NOT_IN_GRID_ERROR} 10.0, 11.0, 12.0"
                ),
            },
            {
                "status": ObservationStatus.MISSING_RESPONSE,
                "missing_realizations": (
                    "0: expected zone 'zone_1' did not match any "
                    "of the simulated zones: zone_2"
                ),
            },
        ]
    )

    with caplog.at_level(logging.INFO):
        UpdateRunModel._log_rft_observations_outside_grid(model, prior, snapshot)

    assert (
        "Update step 2: 1 of 4 RFT observations deactivated because their "
        "location was outside the grid" in caplog.text
    )


def test_that_no_rft_grid_log_is_emitted_without_rft_observations(caplog):
    model = MagicMock(spec=UpdateRunModel)

    prior = MagicMock()
    prior.experiment.observations = {}

    with caplog.at_level(logging.INFO):
        UpdateRunModel._log_rft_observations_outside_grid(model, prior, MagicMock())

    assert "outside the grid" not in caplog.text
