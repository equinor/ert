from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.analysis import (
    ErtAnalysisError,
    UpdateConfiguration,
    smoother_update,
)
from ert.analysis._es_update import (
    TempStorage,
    UpdateSettings,
    _create_temporary_parameter_storage,
)
from ert.analysis.configuration import UpdateStep
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.config import AnalysisConfig, ErtConfig, GenDataConfig, GenKwConfig
from ert.config.analysis_module import ESSettings
from ert.storage import open_storage
from ert.storage.realization_storage_state import RealizationStorageState


@pytest.fixture
def update_config():
    return UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=["PARAMETER"],
            )
        ]
    )


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )


@pytest.fixture
def obs():
    return xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0, 1.0, 1.0]]),
            "std": (["report_step", "index"], [[0.1, 1.0, 10.0]]),
        },
        coords={"index": [0, 1, 2], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )


@pytest.mark.scheduler
@pytest.mark.integration_test
def test_that_posterior_has_lower_variance_than_prior(
    copy_case, try_queue_and_scheduler, monkeypatch
):
    copy_case("poly_example")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "default",
            "--target-case",
            "target",
            "--realizations",
            "1-50",
            "poly.ert",
        ],
    )

    run_cli(parsed)
    facade = LibresFacade.from_config_file("poly.ert")
    with open_storage(facade.enspath) as storage:
        default_fs = storage.get_ensemble_by_name("default")
        df_default = facade.load_all_gen_kw_data(default_fs)
        target_fs = storage.get_ensemble_by_name("target")
        df_target = facade.load_all_gen_kw_data(target_fs)

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_target.cov().to_numpy())
        < np.linalg.det(df_default.cov().to_numpy())
    )


@pytest.mark.scheduler
@pytest.mark.integration_test
def test_that_surfaces_retain_their_order_when_loaded_and_saved_by_ert(
    copy_case, try_queue_and_scheduler, monkeypatch
):
    """This is a regression test to make sure ert does not use the wrong order
    (row-major / column-major) when working with surfaces.
    """
    rng = np.random.default_rng()
    import xtgeo
    from scipy.ndimage import gaussian_filter

    def sample_prior(nx, ny):
        return np.exp(
            5
            * gaussian_filter(
                gaussian_filter(rng.random(size=(nx, ny)), sigma=2.0), sigma=1.0
            )
        )

    copy_case("snake_oil_field")

    nx = 5
    ny = 7
    ensemble_size = 2

    Path("./surface").mkdir()
    for i in range(ensemble_size):
        surf = xtgeo.RegularSurface(
            ncol=nx, nrow=ny, xinc=1.0, yinc=1.0, values=sample_prior(nx, ny)
        )
        surf.to_file(f"surface/surf_init_{i}.irap", fformat="irap_ascii")

    # Single observation with a large ERROR to make sure the udpate is minimal.
    obs = """
    SUMMARY_OBSERVATION WOPR_OP1_9
    {
        VALUE   = 0.1;
        ERROR   = 200.0;
        DATE    = 2010-03-31;
        KEY     = WOPR:OP1;
    };
    """

    with open("observations/observations.txt", "w", encoding="utf-8") as file:
        file.write(obs)

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "snake_oil_surface.ert",
            "--target-case",
            "es_udpate",
        ],
    )
    run_cli(parsed)

    ert_config = ErtConfig.from_file("snake_oil_surface.ert")

    storage = open_storage(ert_config.ens_path)

    ens_prior = storage.get_ensemble_by_name("default")
    ens_posterior = storage.get_ensemble_by_name("es_udpate")

    # Check that surfaces defined in INIT_FILES are not changed by ERT
    surf_prior = ens_prior.load_parameters("TOP", list(range(ensemble_size)))["values"]
    for i in range(ensemble_size):
        _prior_init = xtgeo.surface_from_file(
            f"surface/surf_init_{i}.irap", fformat="irap_ascii", dtype=np.float32
        )
        np.testing.assert_array_equal(surf_prior[i], _prior_init.values.data)

    surf_posterior = ens_posterior.load_parameters("TOP", list(range(ensemble_size)))[
        "values"
    ]

    assert surf_prior.shape == surf_posterior.shape

    for i in range(ensemble_size):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(surf_prior[i], surf_posterior[i])
        np.testing.assert_almost_equal(
            surf_prior[i].values, surf_posterior[i].values, decimal=3
        )


@pytest.mark.scheduler
@pytest.mark.integration_test
def test_update_multiple_param(copy_case, try_queue_and_scheduler, monkeypatch):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    copy_case("snake_oil_field")
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "snake_oil.ert",
            "--target-case",
            "posterior",
        ],
    )

    run_cli(parsed)

    ert_config = ErtConfig.from_file("snake_oil.ert")

    storage = open_storage(ert_config.ens_path)
    sim_fs = storage.get_ensemble_by_name("default")
    posterior_fs = storage.get_ensemble_by_name("posterior")

    def _load_parameters(source_ens, iens_active_index, param_groups):
        temp_storage = TempStorage()
        for param_group in param_groups:
            _temp_storage = _create_temporary_parameter_storage(
                source_ens, iens_active_index, param_group
            )
            temp_storage[param_group] = _temp_storage[param_group]
        return temp_storage

    sim_fs.load_parameters("SNAKE_OIL_PARAM_BPR")["values"]
    param_groups = list(sim_fs.experiment.parameter_configuration.keys())
    prior = _load_parameters(sim_fs, list(range(10)), param_groups)
    posterior = _load_parameters(posterior_fs, list(range(10)), param_groups)

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    # https://en.wikipedia.org/wiki/Variance#For_vector-valued_random_variables
    for prior_name, prior_data in prior.items():
        assert np.trace(np.cov(posterior[prior_name])) < np.trace(np.cov(prior_data))


@pytest.mark.integration_test
def test_gen_data_obs_data_mismatch(storage, uniform_parameter, update_config):
    resp = GenDataConfig(name="RESPONSE")
    obs = xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0]]),
            "std": (["report_step", "index"], [[0.1]]),
        },
        coords={"index": [1000], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        prior.state_map[iens] = RealizationStorageState.HAS_DATA
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        data = rng.uniform(0.8, 1, 3)
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    AnalysisConfig()
    with pytest.raises(
        ErtAnalysisError,
        match="No active observations",
    ):
        smoother_update(
            prior, posterior_ens, "id", update_config, UpdateSettings(), ESSettings()
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_gen_data_missing(storage, update_config, uniform_parameter, obs):
    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        prior.state_map[iens] = RealizationStorageState.HAS_DATA
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        data = rng.uniform(0.8, 1, 2)  # Importantly, shorter than obs
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    update_snapshot = smoother_update(
        prior, posterior_ens, "id", update_config, UpdateSettings(), ESSettings()
    )
    assert [
        step.status for step in update_snapshot.update_step_snapshots["ALL_ACTIVE"]
    ] == ["Active", "Active", "Deactivated, missing response(es)"]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_update_subset_parameters(storage, uniform_parameter, obs):
    no_update_param = GenKwConfig(
        name="EXTRA_PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file=None,
    )
    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter, no_update_param],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        prior.state_map[iens] = RealizationStorageState.HAS_DATA
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        prior.save_parameters(
            "EXTRA_PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )

        data = rng.uniform(0.8, 1, 10)
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    update_config = UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="NOT_ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=["PARAMETER"],  # No EXTRA_PARAMETER here
            )
        ]
    )
    smoother_update(
        prior, posterior_ens, "id", update_config, UpdateSettings(), ESSettings()
    )
    assert prior.load_parameters("EXTRA_PARAMETER", 0)["values"].equals(
        posterior_ens.load_parameters("EXTRA_PARAMETER", 0)["values"]
    )
    assert not prior.load_parameters("PARAMETER", 0)["values"].equals(
        posterior_ens.load_parameters("PARAMETER", 0)["values"]
    )
