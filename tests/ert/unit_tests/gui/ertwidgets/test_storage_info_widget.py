from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from PyQt6.QtWidgets import (
    QTreeWidget,
)

from ert.config import ErtConfig, ObservationType
from ert.config.rft_config import RFTConfig
from ert.gui.tools.manage_experiments.storage_info_widget import (
    _EnsembleWidget,
    _EnsembleWidgetTabs,
)


def create_experiment_from_config(config: ErtConfig, storage):
    ens_config = config.ensemble_config

    def dump_all(configurations):
        return [c.model_dump(mode="json") for c in configurations]

    experiment = storage.create_experiment(
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
    return experiment


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
                    {
                        "type": ObservationType.SUMMARY,
                        "name": "sumobs",
                        "KEY": observation_key,
                        "DATE": date.isoformat(),
                        "VALUE": 1.0,
                        "ERROR": 1.0,
                    }
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
                    {
                        "type": ObservationType.BREAKTHROUGH,
                        "name": "BRT_OP1",
                        "KEY": key,
                        "ERROR": "3",
                        "DATE": (date + timedelta(days=5)).isoformat(),
                        "THRESHOLD": 0.4,
                    }
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
                "well_connection_cell": pl.Series([cell1], dtype=pl.Array(pl.Int64, 3)),
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
    ensemble.save_response("rft", rft_response(), 0)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones1, well_connection_cell=cell1), 0
    )

    # realization 1: matching response
    ensemble.save_response("rft", rft_response(), 1)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones2, well_connection_cell=cell1), 1
    )

    # realization 2: disabled due to both zones and well connection cell mismatch
    ensemble.save_response("rft", rft_response(), 2)
    ensemble.save_observation_location_metadata(
        location_metadata(actual_zones=zones1, well_connection_cell=cell2), 2
    )

    # realization 3: disabled due to well connection cell mismatch
    ensemble.save_response("rft", rft_response(), 3)
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
    date = datetime(year=2000, month=1, day=1).date()  # noqa: DTZ001
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "SUMMARY": ["*"],
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
                        "ZONE": "zone",
                    },
                    {
                        "type": ObservationType.RFT,
                        "name": "RFT_index_key_duplicate",
                        "WELL": "WELL",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": date.isoformat(),
                        "PROPERTY": "PRESSURE",
                        "EAST": 10.0,
                        "NORTH": 11.0,
                        "TVD": 12.0,
                        "ZONE": "zone",
                    },
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
                {
                    "type": ObservationType.BREAKTHROUGH,
                    "name": "BRT_OP1",
                    "KEY": "WWCT:OP1",
                    "ERROR": "3",
                    "DATE": datetime(year=2000, month=1, day=1).isoformat(),  # noqa: DTZ001
                    "THRESHOLD": 0.4,
                },
                {
                    "type": ObservationType.BREAKTHROUGH,
                    "name": "BRT_OP2",
                    "KEY": "WWCT:OP1",
                    "ERROR": "3",
                    "DATE": datetime(year=2000, month=1, day=9).isoformat(),  # noqa: DTZ001
                    "THRESHOLD": 0.7,
                },
            ],
            ["0.4", "0.7"],
            id="breakthrough",
        ),
        pytest.param(
            [
                {
                    "type": ObservationType.RFT,
                    "name": "RFT2",
                    "WELL": "WELL",
                    "VALUE": "700",
                    "ERROR": "0.1",
                    "DATE": "2000-01-01",
                    "PROPERTY": "PRESSURE",
                    "EAST": 11.0,
                    "NORTH": 5.0,
                    "TVD": 4.0,
                    "ZONE": "zone",
                },
                {
                    "type": ObservationType.RFT,
                    "name": "RFT1",
                    "WELL": "WELL",
                    "VALUE": "700",
                    "ERROR": "0.1",
                    "DATE": "2000-01-01",
                    "PROPERTY": "PRESSURE",
                    "EAST": 5.0,
                    "NORTH": 6.0,
                    "TVD": 7.0,
                    "ZONE": "zone",
                },
            ],
            ["5.0, 6.0, 7.0, zone", "11.0, 5.0, 4.0, zone"],
            id="rft",
        ),
        pytest.param(
            [
                {
                    "type": ObservationType.RFT,
                    "name": "RFT2",
                    "WELL": "WELL",
                    "VALUE": "700",
                    "ERROR": "0.1",
                    "DATE": "2000-01-01",
                    "PROPERTY": "PRESSURE",
                    "EAST": 700.0,
                    "NORTH": 500.0,
                    "TVD": 400.0,
                },
                {
                    "type": ObservationType.RFT,
                    "name": "RFT1",
                    "WELL": "WELL",
                    "VALUE": "700",
                    "ERROR": "0.1",
                    "DATE": "2000-01-01",
                    "PROPERTY": "PRESSURE",
                    "EAST": 5.0,
                    "NORTH": 6.0,
                    "TVD": 7.0,
                },
            ],
            ["5.0, 6.0, 7.0, None", "700.0, 500.0, 400.0, None"],
            id="rft without zone",
        ),
        pytest.param(
            [
                {
                    "type": ObservationType.SUMMARY,
                    "name": "FOPR_1",
                    "KEY": "FOPR",
                    "VALUE": "1",
                    "ERROR": "1",
                    "DATE": "2023-03-15",
                },
                {
                    "type": ObservationType.SUMMARY,
                    "name": "FOPR_2",
                    "KEY": "FOPR",
                    "VALUE": "1",
                    "ERROR": "1",
                    "DATE": "2024-11-07",
                },
            ],
            ["2023-03-15", "2024-11-07"],
            id="summary",
        ),
        pytest.param(
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "GOBS1",
                    "DATA": "GEN",
                    "RESTART": "11",
                    "INDEX_LIST": "60,400,1200,1600,1800",
                    "OBS_FILE": "<OBS_FILE_PLACEHOLDER>",
                },
                {
                    "type": ObservationType.GENERAL,
                    "name": "GOBS2",
                    "DATA": "GEN",
                    "RESTART": "100",
                    "INDEX_LIST": "60,400,1200,1600,1800",
                    "OBS_FILE": "<OBS_FILE_PLACEHOLDER>",
                },
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
    ],
)
def test_that_observations_are_identified_and_sorted_by_full_index_key(
    qtbot, storage, tmp_path, observations, expected_name_order
):
    def patch_gen_obs_file(observations_list):
        obs_file = tmp_path / "obs_data.txt"
        gen_obs = [float(i + 1) for i in range(5)]
        obs_file.write_text(
            "\n".join(f"{obs} {obs}" for obs in gen_obs),
            encoding="utf-8",
        )

        return [
            {
                k: str(obs_file) if v == "<OBS_FILE_PLACEHOLDER>" else v
                for k, v in observation_dict.items()
            }
            for observation_dict in observations_list
        ]

    observations = patch_gen_obs_file(observations)
    config = ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASE",
            "GEN_DATA": [
                ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": "100,11"}]
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
