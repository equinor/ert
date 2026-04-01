from datetime import datetime, timedelta

import polars as pl
import pytest
from PyQt6.QtWidgets import (
    QTreeWidget,
)

from ert.config import ErtConfig, ObservationType
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


def test_that_missing_response_for_observation_response_key_does_not_crash(
    qtbot, storage
):
    date = datetime(year=2000, month=1, day=1)
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


def test_that_breakthrough_experiment_does_not_crash(qtbot, storage):
    date = datetime(year=2000, month=1, day=1)
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


def test_that_rft_experiment_without_a_zone_does_not_crash(qtbot, storage):
    date = datetime(year=2000, month=1, day=1).date()
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

    def partial_rft_response() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "response_key": ["WELL:2000-01-01:PRESSURE"],
                "well": ["WELL"],
                "date": [date.isoformat()],
                "property": ["PRESSURE"],
                "time": [date],
                "depth": [0.0],
                "values": [0.0],
                "zone": None,
                "east": [10.0],
                "north": [11.0],
                "tvd": [12.0],
            }
        )

    ensemble.save_response("rft", partial_rft_response(), 0)

    ensemble_widget = _EnsembleWidget()
    ensemble_widget.setEnsemble(ensemble)
    qtbot.addWidget(ensemble_widget)

    panels_widget = ensemble_widget._tab_widget
    panels_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    plot = ensemble_widget._figure.get_axes()
    assert len(plot) == 1
    assert len(plot[0].collections) == 2


def test_that_both_observations_with_same_data_are_displayed(qtbot, storage):
    date = datetime(year=2000, month=1, day=1).date()
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
                        "name": "RFT_match_key_duplicate",
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
                    "DATE": datetime(year=2000, month=1, day=1).isoformat(),
                    "THRESHOLD": 0.4,
                },
                {
                    "type": ObservationType.BREAKTHROUGH,
                    "name": "BRT_OP2",
                    "KEY": "WWCT:OP1",
                    "ERROR": "3",
                    "DATE": datetime(year=2000, month=1, day=9).isoformat(),
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
def test_that_observations_are_identified_and_sorted_by_full_match_key(
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
