from unittest.mock import MagicMock

from ert.gui.ertwidgets.chip import DataSource
from ert.gui.experiments import ExperimentPanel


def test_that_summary_shows_parameters_and_observations_of_the_selected_ensemble():
    stored_parameter = MagicMock()
    ensemble = MagicMock()
    ensemble.name = "iter-0"
    ensemble.experiment.name = "my_experiment"
    ensemble.experiment.parameter_configuration = {"stored": stored_parameter}
    ensemble.experiment.experiment_config = {
        "observations": [
            {"name": "OBS", "type": "summary_observation"},
            {"name": "OBS", "type": "summary_observation"},
        ]
    }
    config_panel = MagicMock()
    config_panel.get_parameter_configuration.return_value = None
    config_panel.get_selected_ensemble.return_value = ensemble
    experiment_panel = MagicMock()
    experiment_panel._experiment_widgets = {
        experiment_panel.get_current_experiment_type(): config_panel
    }

    ExperimentPanel._update_parameter_summary(experiment_panel)

    summary = experiment_panel.configuration_summary
    summary.set_parameter_configuration.assert_called_once_with(
        [stored_parameter], DataSource.SELECTED_ENSEMBLE, "my_experiment : iter-0"
    )
    summary.set_observations.assert_called_once_with(
        [{"observation_key": "OBS", "count": 2}],
        DataSource.SELECTED_ENSEMBLE,
        "my_experiment : iter-0",
    )
