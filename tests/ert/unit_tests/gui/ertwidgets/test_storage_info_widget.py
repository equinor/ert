from datetime import datetime, timedelta
from textwrap import dedent

import numpy as np
import polars as pl
import pytest
from PyQt6.QtWidgets import (
    QTreeWidget,
)

from ert.config import ErtConfig, ObservationType
from ert.config.rft_config import RFTConfig
from ert.ensemble_evaluator import state
from ert.gui.experiments.view import run_status as run_status_module
from ert.gui.experiments.view.realization import RealizationWidget
from ert.gui.experiments.view.run_status import RunStatusView
from ert.gui.tools.manage_experiments.storage_info_widget import (
    _EnsembleWidget,
    _EnsembleWidgetTabs,
)
from ert.run_models.event import FullSnapshotEvent, status_event_to_json
from tests.ert.defaults_generator import (
    create_breakthrough_observation_dict,
    create_general_observation_dict,
    create_rft_observation_dict,
    create_seismic_observation_dict,
    create_summary_observation_dict,
)
from tests.ert.utils import SnapshotBuilder


def create_experiment_from_config(config: ErtConfig, storage):
    ens_config = config.ensemble_config

    def dump_all(configurations):
        return [c.model_dump(mode="json") for c in configurations]

    return storage.create_experiment(
        experiment_config={
            "parameter_configuration": dump_all(ens_config.parameter_configuration),
            "response_configuration": dump_all(ens_config.response_configuration),
            "derived_response_configuration": dump_all(
                ens_config.derived_response_configuration
            ),
            "observations": dump_all(config.observation_declarations),
            "ert_templates": config.ert_templates,
        },
    )


@pytest.mark.filterwarnings("ignore:.*contains a SUMMARY key but no forward model step")
def test_that_missing_response_for_observation_response_key_does_not_crash(
    qtbot, storage
):
    date = datetime(year=2000, month=1, day=1)  # noqa: DTZ001
    observation_key = "FOPR"
    requested_keys = ["*"]
    received_keys = ["WRONG"]

    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "SUMMARY": requested_keys,
            "OBS_CONFIG": (
                "obs_config",
                [
                    create_summary_observation_dict(
                        key=observation_key, date=date.isoformat()
                    ),
                ],
            ),
        }
    )

    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)
    ensemble.save_response(
        "summary",
        pl.DataFrame(
            {
                "response_key": received_keys,
                "time": [pl.Series([date]).dt.cast_time_unit("ms")],
                "values": [pl.Series([1.0], dtype=pl.Float32)],
            }
        ).explode("values", "time"),
        0,
    )

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    assert len(ensemble_widget._figure.get_axes()) == 1


@pytest.mark.filterwarnings("ignore:.*contains a SUMMARY key but no forward model step")
def test_that_breakthrough_experiment_does_not_crash(qtbot, storage):
    date = datetime(year=2000, month=1, day=1)  # noqa: DTZ001
    key = "WWCT:OP1"

    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "OBS_CONFIG": (
                "obs_config",
                [
                    create_breakthrough_observation_dict(key=key, date=date),
                ],
            ),
        }
    )
    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)

    def summary_response() -> pl.DataFrame:
        num_points = 15
        values = range(num_points)

        return pl.DataFrame(
            {
                "response_key": [key] * num_points,
                "time": [date + timedelta(days=day) for day in range(num_points)],
                "values": pl.Series(values, dtype=pl.Float32),
            }
        )

    ensemble.save_response("summary", summary_response(), 0)
    bt_config = experiment.derived_response_configuration["breakthrough"]
    breakthrough_response = bt_config.derive_from_storage(0, 0, ensemble)
    ensemble.save_response("breakthrough", breakthrough_response, 0)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    assert len(ensemble_widget._figure.get_axes()) == 1


@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_rft_experiment_without_a_zone_does_not_crash(qtbot, storage):
    date = datetime(year=2000, month=1, day=1).date()  # noqa: DTZ001
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "RFT": [{"WELL": "WELL", "DATE": "2000-01-01", "PROPERTIES": "PRESSURE"}],
            "OBS_CONFIG": (
                "obs_config",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "RFT",
                        "WELL": "WELL",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": date.isoformat(),
                        "PROPERTY": "PRESSURE",
                        "EAST": 10.0,
                        "NORTH": 11.0,
                        "TVD": 12.0,
                    },
                ],
            ),
        }
    )
    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)

    def rft_response() -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "response_key": ["WELL:2000-01-01:PRESSURE"],
                "well": ["WELL"],
                "date": [date.isoformat()],
                "property": ["PRESSURE"],
                "time": [date],
                "depth": [0.0],
                "values": [0.0],
                "well_connection_cell": pl.Series(
                    [[7, 7, 8]], dtype=pl.Array(pl.Int64, 3)
                ),
                "cell_center": pl.Series(
                    [[np.nan, np.nan, np.nan]], dtype=pl.Array(pl.Float32, 3)
                ),
                "cell_zones": [["zone100"]],
            },
            schema=RFTConfig.response_schema(),
        )
        RFTConfig._assert_schema(df, RFTConfig.response_schema())
        return df

    def location_metadata() -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "east": [10.0],
                "north": [11.0],
                "tvd": [12.0],
                "actual_zones": [["zone100"]],
                "well_connection_cell": pl.Series(
                    [[7, 7, 8]], dtype=pl.Array(pl.Int64, 3)
                ),
                "well_connection_cell_center": pl.Series(
                    [[10.0, 11.0, 12.0]], dtype=pl.Array(pl.Float32, 3)
                ),
            },
            schema=RFTConfig.location_metadata_schema(),
        )
        RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())
        return df

    ensemble.save_response("rft", rft_response(), 0)
    ensemble.save_observation_location_metadata(location_metadata(), 0)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    plot = ensemble_widget._figure.get_axes()
    assert len(plot) == 1
    assert len(plot[0].collections) == 2


@pytest.mark.parametrize(
    ("approximate_missing_rft_values", "visualized_responses"), [(True, 1), (False, 0)]
)
@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_approximated_rft_responses_are_visualized_in_ensemble_widget_if_enabled(
    qtbot, storage, approximate_missing_rft_values, visualized_responses
):
    date = datetime(year=2000, month=1, day=1).date()  # noqa: DTZ001
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "RFT": [{"WELL": "WELL", "DATE": "2000-01-01", "PROPERTIES": "PRESSURE"}],
            "APPROXIMATE_MISSING_RFT_VALUES": approximate_missing_rft_values,
            "OBS_CONFIG": (
                "obs_config",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "RFT_approx",
                        "WELL": "WELL",
                        "VALUE": "150",
                        "ERROR": "0.1",
                        "DATE": date.isoformat(),
                        "PROPERTY": "PRESSURE",
                        "EAST": 0.0,
                        "NORTH": 0.0,
                        "TVD": 2.0,
                        "ZONE": "zone1",
                    },
                ],
            ),
        }
    )
    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)

    # Two responses bracketing the observation location along the z-axis in zone1.
    # Cell centers are at (0, 0, 1) and (0, 0, 3); the observation is at tvd=2,
    # so it projects onto the midpoint (t=0.5), giving an interpolated value of 150.
    rft_responses = pl.DataFrame(
        {
            "response_key": [
                "WELL:2000-01-01:PRESSURE",
                "WELL:2000-01-01:PRESSURE",
            ],
            "well": ["WELL", "WELL"],
            "date": [date.isoformat(), date.isoformat()],
            "property": ["PRESSURE", "PRESSURE"],
            "time": [date, date],
            "depth": pl.Series([1.0, 3.0], dtype=pl.Float32),
            "values": pl.Series([100.0, 200.0], dtype=pl.Float32),
            "well_connection_cell": pl.Series(
                [[1, 1, 1], [1, 1, 3]], dtype=pl.Array(pl.Int64, 3)
            ),
            "cell_center": pl.Series(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=pl.Array(pl.Float32, 3)
            ),
            "cell_zones": [["zone1"], ["zone1"]],
        },
        schema=RFTConfig.response_schema(),
    )

    # The observation maps to cell [1, 1, 2], which has no corresponding response.
    location_metadata = pl.DataFrame(
        {
            "east": [0.0],
            "north": [0.0],
            "tvd": [2.0],
            "actual_zones": [["zone1"]],
            "well_connection_cell": pl.Series([[1, 1, 2]], dtype=pl.Array(pl.Int64, 3)),
            "well_connection_cell_center": pl.Series(
                [[0.0, 0.0, 2.0]], dtype=pl.Array(pl.Float32, 3)
            ),
        },
        schema=RFTConfig.location_metadata_schema(),
    )

    ensemble.save_response("rft", rft_responses, 0)
    ensemble.save_observation_location_metadata(location_metadata, 0)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    plot = ensemble_widget._figure.get_axes()
    assert len(plot) == 1
    collections = plot[0].collections
    if visualized_responses > 0:
        assert len(collections) == 2
        strip_plot_response_collection = collections[-1]
        displayed_responses = np.asarray(strip_plot_response_collection.get_offsets())
        assert len(displayed_responses) == visualized_responses
    else:
        # Without approximation no responses match, so only the observation
        # errorbar is rendered and no response strip plot is added.
        assert len(collections) == 1


@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_many_realizations_in_rft_affect_responses_not_observation_tree(
    qtbot, storage
):
    date = datetime(year=2000, month=1, day=1).date()  # noqa: DTZ001
    zone_a = "zone_a"
    zone_b = "zone_b"
    zones1 = [zone_a]
    zones2 = [zone_a, zone_b]
    cell1 = [7, 7, 8]
    cell2 = [7, 7, 9]
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "RFT": [{"WELL": "WELL", "DATE": "2000-01-01", "PROPERTIES": "PRESSURE"}],
            "OBS_CONFIG": (
                "obs_config",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "RFT1",
                        "WELL": "WELL",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": date.isoformat(),
                        "PROPERTY": "PRESSURE",
                        "EAST": 10.0,
                        "NORTH": 11.0,
                        "TVD": 12.0,
                        "ZONE": zone_b,
                    },
                    {
                        "type": ObservationType.RFT,
                        "name": "RFT2",
                        "WELL": "WELL",
                        "VALUE": "800",
                        "ERROR": "0.1",
                        "DATE": date.isoformat(),
                        "PROPERTY": "PRESSURE",
                        "EAST": 15.0,
                        "NORTH": 16.0,
                        "TVD": 17.0,
                        "ZONE": zone_b,
                    },
                ],
            ),
        }
    )

    def rft_response(
        *,
        cell_zones: list[str],
    ) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "response_key": ["WELL:2000-01-01:PRESSURE"],
                "well": ["WELL"],
                "date": [date.isoformat()],
                "property": ["PRESSURE"],
                "time": [date],
                "depth": [0.0],
                "values": [0.0],
                "well_connection_cell": pl.Series([cell1], dtype=pl.Array(pl.Int64, 3)),
                "cell_center": pl.Series(
                    [[np.nan, np.nan, np.nan]], dtype=pl.Array(pl.Float32, 3)
                ),
                "cell_zones": [cell_zones],
            },
            schema=RFTConfig.response_schema(),
        )
        RFTConfig._assert_schema(df, RFTConfig.response_schema())
        return df

    def location_metadata(
        *,
        actual_zones: list[str],
        well_connection_cell: list[int],
    ) -> pl.DataFrame:
        df = pl.DataFrame(
            [
                {
                    "east": 10.0,
                    "north": 11.0,
                    "tvd": 12.0,
                    "actual_zones": actual_zones,
                    "well_connection_cell": well_connection_cell,
                },
                {
                    "east": 15.0,
                    "north": 16.0,
                    "tvd": 17.0,
                    "actual_zones": actual_zones,
                    "well_connection_cell": well_connection_cell,
                },
            ],
            schema=RFTConfig.location_metadata_schema(),
        )
        RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())
        return df

    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=4)

    # realization 0: disabled due to zone mismatch
    ensemble.save_response("rft", rft_response(cell_zones=zones1), 0)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones1, well_connection_cell=cell1), 0
    )

    # realization 1: matching response
    ensemble.save_response("rft", rft_response(cell_zones=zones2), 1)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones2, well_connection_cell=cell1), 1
    )

    # realization 2: disabled due to both zones and well connection cell mismatch
    ensemble.save_response("rft", rft_response(cell_zones=zones1), 2)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones1, well_connection_cell=cell2), 2
    )

    # realization 3: disabled due to well connection cell mismatch
    ensemble.save_response("rft", rft_response(cell_zones=zones2), 3)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones2, well_connection_cell=cell2), 3
    )

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    observation_widget = ensemble_widget.findChild(QTreeWidget)
    top = observation_widget.topLevelItem(0)
    assert top

    def verify_observation_tree():
        assert top.childCount() == 2
        children = [
            child.text(0)
            for i in range(top.childCount())
            if (child := top.child(i)) is not None
        ]
        expected_children = ["10.0, 11.0, 12.0, zone_b", "15.0, 16.0, 17.0, zone_b"]
        assert children == expected_children, (
            f"Expected observation names {expected_children}, but got {children}"
        )

    def verify_number_of_visualized_responses():
        plot = ensemble_widget._figure.get_axes()
        assert len(plot) == 1
        collections = plot[0].collections
        assert len(collections) == 2
        strip_plot_response_collection = collections[-1]
        displayed_responses = np.asarray(strip_plot_response_collection.get_offsets())
        assert len(displayed_responses) == 1

    verify_observation_tree()
    observation_widget.setCurrentItem(top.child(0))
    verify_number_of_visualized_responses()
    observation_widget.setCurrentItem(top.child(1))
    verify_number_of_visualized_responses()


@pytest.mark.filterwarnings("ignore:.*contains a SUMMARY key but no forward model step")
def test_that_both_observations_with_same_data_are_displayed(qtbot, storage):
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "SUMMARY": ["*"],
            "OBS_CONFIG": (
                "obs_config",
                [
                    create_rft_observation_dict(name="RFT"),
                    create_rft_observation_dict(name="RFT_index_key_duplicate"),
                ],
            ),
        }
    )
    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    observation_widget = ensemble_widget.findChild(QTreeWidget)
    top = observation_widget.topLevelItem(0)
    assert top
    assert top.childCount() == 2


@pytest.mark.filterwarnings("ignore:.*contains a SUMMARY key but no forward model step")
@pytest.mark.parametrize(
    ("observations", "expected_name_order"),
    [
        pytest.param(
            [
                create_breakthrough_observation_dict(
                    name="BRT_OP1",
                    threshold=0.4,
                    date=datetime(year=2000, month=1, day=1),  # noqa: DTZ001
                ),
                create_breakthrough_observation_dict(
                    name="BRT_OP2",
                    threshold=0.7,
                    date=datetime(year=2000, month=1, day=9),  # noqa: DTZ001
                ),
            ],
            ["0.4", "0.7"],
            id="breakthrough",
        ),
        pytest.param(
            [
                create_rft_observation_dict(
                    name="RFT2", east=11.0, north=5.0, tvd=4.0, zone="zone"
                ),
                create_rft_observation_dict(
                    name="RFT1", east=5.0, north=6.0, tvd=7.0, zone="zone"
                ),
            ],
            ["5.0, 6.0, 7.0, zone", "11.0, 5.0, 4.0, zone"],
            id="rft",
        ),
        pytest.param(
            [
                create_rft_observation_dict(
                    name="RFT2", east=700.0, north=500.0, tvd=400.0
                ),
                create_rft_observation_dict(name="RFT1", east=5.0, north=6.0, tvd=7.0),
            ],
            ["5.0, 6.0, 7.0, None", "700.0, 500.0, 400.0, None"],
            id="rft without zone",
        ),
        pytest.param(
            [
                create_summary_observation_dict(name="FOPR_1", date="2023-03-15"),
                create_summary_observation_dict(name="FOPR_2", date="2024-11-07"),
            ],
            ["2023-03-15", "2024-11-07"],
            id="summary",
        ),
        pytest.param(
            [
                create_general_observation_dict(
                    name="GOBS1",
                    data="GEN",
                    restart=11,
                    index_list="60,400,1200,1600,1800",
                    obs_file="gen_obs_data.txt",
                ),
                create_general_observation_dict(
                    name="GOBS2",
                    data="GEN",
                    restart=100,
                    index_list="60,400,1200,1600,1800",
                    obs_file="gen_obs_data.txt",
                ),
            ],
            [
                "11, 60",
                "11, 400",
                "11, 1200",
                "11, 1600",
                "11, 1800",
                "100, 60",
                "100, 400",
                "100, 1200",
                "100, 1600",
                "100, 1800",
            ],
            id="general",
        ),
        pytest.param(
            [
                create_seismic_observation_dict(
                    name="SEISMIC1", csv="seismic--20250101_20240101.csv"
                ),
                create_seismic_observation_dict(
                    name="SEISMIC2", csv="seismic--20260101_20240101.csv"
                ),
            ],
            ["99.0, 200.0", "100.0, 200.0"],
            id="seismic",
        ),
    ],
)
def test_that_observations_are_identified_and_sorted_by_full_index_key(
    qtbot, storage, observations, expected_name_order, mocked_files
):

    def mock_gen_obs_file():
        gen_obs_file = "gen_obs_data.txt"
        gen_obs = [float(i + 1) for i in range(5)]
        gen_obs_data = "\n".join(f"{obs} {obs}" for obs in gen_obs)
        mocked_files[gen_obs_file] = gen_obs_data

    def mock_seismic_obs_file():
        seismic_obs_file_1 = "seismic--20250101_20240101.csv"
        seismic_obs_file_2 = "seismic--20260101_20240101.csv"
        seismic_obs_data = dedent(
            """
            X_UTME,Y_UTMN,OBS,OBS_ERROR,REGION
            99.00,200.00,1.0,0.005,1.0
            100.00,200.00,2.0,0.005,1.0
            """
        )
        mocked_files[seismic_obs_file_1] = seismic_obs_data
        mocked_files[seismic_obs_file_2] = seismic_obs_data

    mock_gen_obs_file()
    mock_seismic_obs_file()

    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "GEN_DATA": [
                ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": "100,11"}]
            ],
            "SEISMIC": [
                "seismic--20250101_20240101.csv",
                "seismic--20260101_20240101.csv",
            ],
            "OBS_CONFIG": (
                "obs_config",
                observations,
            ),
        }
    )

    experiment = create_experiment_from_config(config, storage)
    ensemble = experiment.create_ensemble(name="default", ensemble_size=1)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    observation_widget = ensemble_widget.findChild(QTreeWidget)
    top = observation_widget.topLevelItem(0)
    assert top
    children = [
        child.text(0)
        for i in range(top.childCount())
        if (child := top.child(i)) is not None
    ]
    assert children == expected_name_order, (
        f"Expected observation names in order {expected_name_order}, but got {children}"
    )


def _build_run_status_snapshot(iteration: int) -> FullSnapshotEvent:
    return FullSnapshotEvent(
        snapshot=SnapshotBuilder()
        .add_fm_step(
            fm_step_id="0",
            index="0",
            name="fm_step_0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
        )
        .build(
            real_ids=["0", "1"],
            status=state.REALIZATION_STATE_FINISHED,
        ),
        iteration_label=f"Running forecast for iteration: {iteration}",
        total_iterations=1,
        progress=1.0,
        realization_count=2,
        status_count={"Finished": 2},
        iteration=iteration,
    )


def _persist_full_snapshot(experiment, iteration: int) -> None:
    event = _build_run_status_snapshot(iteration)
    path = experiment.status_snapshot_path(iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status_event_to_json(event), encoding="utf-8")


def test_that_run_status_tab_shows_persisted_realizations_for_ensemble(qtbot, storage):
    experiment = storage.create_experiment(name="exp")
    ensemble = experiment.create_ensemble(name="default", ensemble_size=2, iteration=0)
    _persist_full_snapshot(experiment, ensemble.iteration)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.RUN_STATUS_TAB)

    run_status_view = ensemble_widget._run_status_view
    qtbot.waitUntil(
        lambda: (
            run_status_view._stack.currentWidget() is not run_status_view._placeholder
        ),
        timeout=5000,
    )

    realization_widget = run_status_view.findChild(RealizationWidget)
    assert realization_widget is not None
    assert realization_widget._real_list_model.rowCount() == 2


def test_that_run_status_tab_shows_placeholder_when_no_snapshot_exists(qtbot, storage):
    experiment = storage.create_experiment(name="exp")
    ensemble = experiment.create_ensemble(name="default", ensemble_size=2, iteration=0)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.RUN_STATUS_TAB)

    run_status_view = ensemble_widget._run_status_view
    assert run_status_view._stack.currentWidget() is run_status_view._placeholder


def test_that_run_status_view_reuses_loaded_snapshot_when_path_is_unchanged(
    qtbot, tmp_path, monkeypatch
):
    path = tmp_path / "snapshot_0.json"
    load_call_count = 0

    def load_snapshot(_):
        nonlocal load_call_count
        load_call_count += 1
        return _build_run_status_snapshot(iteration=0)

    monkeypatch.setattr(run_status_module, "load_status_snapshot_event", load_snapshot)

    run_status_view = RunStatusView()
    qtbot.addWidget(run_status_view)

    run_status_view.load_snapshot(path)
    assert load_call_count == 1
    content_after_first_load = run_status_view._content

    run_status_view.load_snapshot(path)

    assert load_call_count == 1
    assert run_status_view._content is content_after_first_load
