from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import xarray as xr

from ert.run_context import RunContext
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

    with patch(
        "ert.run_context.RunContext.active_realizations",
        new_callable=PropertyMock,
    ) as mock_active_realizations:
        mock_active_realizations.return_value = [1, 2, 3, 4]

        prior_context = RunContext(
            ensemble=ensemble,
            runpaths=MagicMock(),
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

        ensemble_experiment._validate_results(prior_context)


def test_validate_results_with_error(snake_oil_default_storage):
    ensemble = snake_oil_default_storage
    active_mask = [True] * 5

    # Modify observations in storage
    ds = xr.open_dataset(
        ensemble.experiment._path / "observations/summary", engine="scipy"
    )
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
    ds_new = xr.merge([ds, ds_fopr2], join="outer")
    ds_new.to_netcdf(ensemble.experiment._path / "observations/summary", engine="scipy")

    with patch(
        "ert.run_context.RunContext.active_realizations",
        new_callable=PropertyMock,
    ) as mock_active_realizations:
        mock_active_realizations.return_value = [1, 2, 3, 4]

        prior_context = RunContext(
            ensemble=ensemble,
            runpaths=MagicMock(),
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
            ensemble_experiment._validate_results(prior_context)
        assert "No active observations for update step" in str(exc.value)
