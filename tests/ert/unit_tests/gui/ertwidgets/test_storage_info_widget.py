from datetime import datetime, timedelta

import polars as pl

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
