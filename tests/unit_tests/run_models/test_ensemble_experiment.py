from unittest.mock import MagicMock

import pandas as pd
import pytest

from ert.run_models import EnsembleExperiment
from ert.run_models.base_run_model import ErtRunError
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
)


@pytest.mark.parametrize(
    "active_mask, expected",
    [
        ([True, True, True, True], True),
        ([False, False, True, False], False),
        ([], False),
        ([False, True, True], True),
        ([False, False, True], False),
    ],
)
def test_check_if_runpath_exists(
    create_dummy_run_path,
    active_mask: list,
    expected: bool,
):
    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        minimum_required_realizations=0,
        ensemble_size=1,
        experiment_name="no-name",
    )

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    EnsembleExperiment.validate = MagicMock()
    ensemble_experiment = EnsembleExperiment(
        simulation_arguments, MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    ensemble_experiment.run_paths.get_paths = get_run_path_mock
    assert ensemble_experiment.check_if_runpath_exists() == expected


def test_validate_results(snake_oil_default_storage):
    ensemble = snake_oil_default_storage
    active_mask = [True] * 5

    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        minimum_required_realizations=1,
        ensemble_size=5,
        experiment_name="no-name",
    )

    EnsembleExperiment.validate = MagicMock()
    ensemble_experiment = EnsembleExperiment(
        simulation_arguments, MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    ensemble_experiment._validate_results(ensemble, active_mask)


def test_validate_results_with_error(snake_oil_default_storage):
    ensemble = snake_oil_default_storage
    active_mask = [True] * 5

    ds_fopr2 = (
        pd.DataFrame(
            data={
                "name": "FOPR_2",
                "obs_name": "FOPR_22,",
                "time": [pd.to_datetime("2010-01-10")],
                "std": [0.2],
                "observations": [2.2],
            }
        )
        .set_index(["name", "obs_name", "time"])
        .to_xarray()
    )
    ds_fopr2.to_netcdf(
        ensemble.experiment._path / "observations/FOPR_2", engine="scipy"
    )

    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        minimum_required_realizations=1,
        ensemble_size=5,
        experiment_name="no-name",
    )

    EnsembleExperiment.validate = MagicMock()
    ensemble_experiment = EnsembleExperiment(
        simulation_arguments, MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    with pytest.raises(ErtRunError) as exc:
        ensemble_experiment._validate_results(ensemble, active_mask)
    assert "No active observations for update step" in str(exc.value)
