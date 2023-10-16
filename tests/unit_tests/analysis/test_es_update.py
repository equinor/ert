import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from iterative_ensemble_smoother import SIES

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.analysis import (
    ErtAnalysisError,
    UpdateConfiguration,
    iterative_smoother_update,
    smoother_update,
)
from ert.analysis._es_update import TempStorage, _create_temporary_parameter_storage
from ert.analysis.configuration import UpdateStep
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.config import AnalysisConfig, ErtConfig, GenDataConfig, GenKwConfig
from ert.realization_state import RealizationState
from ert.storage import open_storage


@pytest.fixture()
def minimal_config(use_tmpdir):
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write("NUM_REALIZATIONS 1")
    ert_config = ErtConfig.from_file("config_file.ert")
    yield ert_config


def remove_timestamp_from_logfile(log_file: Path):
    with open(log_file, "r", encoding="utf-8") as fin:
        buf = fin.read()
    buf = re.sub(
        r"Time: [0-9]{4}\.[0-9]{2}\.[0-9]{2} [0-9]{2}\:[0-9]{2}\:[0-9]{2}", "Time:", buf
    )
    with open(log_file, "w", encoding="utf-8") as fout:
        fout.write(buf)


def test_update_report(snake_oil_case_storage, snake_oil_storage, snapshot):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert.getEnsembleSize(),
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        "id",
        ert.getLocalConfig(),
        ert.analysisConfig(),
    )
    log_file = Path(ert.analysisConfig().log_path) / "id.txt"
    remove_timestamp_from_logfile(log_file)
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")


@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        (
            "IES_ENKF",
            [
                0.515356585450388,
                -0.7997450173495089,
                -0.673803314701884,
                -0.12006348287921552,
                0.12835309068473374,
                0.056452419575246444,
                -1.5161257610231536,
                0.2401457090342254,
                -0.7985453300893501,
                0.7764022070573613,
            ],
        ),
        (
            "STD_ENKF",
            [
                0.4658755223614102,
                0.08294244626646294,
                -1.2728836885070545,
                -0.7044037773899394,
                0.0701040026601418,
                0.25463877762608783,
                -1.7638615728377676,
                1.0900234695729822,
                -1.2135225153906364,
                1.27516244886867,
            ],
        ),
    ],
)
def test_update_snapshot(
    snake_oil_case_storage,
    snake_oil_storage,
    module,
    expected_gen_kw,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    ert.analysisConfig().select_module(module)
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert.getEnsembleSize(),
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )
    if module == "IES_ENKF":
        w_container = SIES(ert.getEnsembleSize())
        iterative_smoother_update(
            prior_ens,
            posterior_ens,
            w_container,
            "id",
            ert.getLocalConfig(),
            ert.analysisConfig(),
            ert.rng(),
        )
    else:
        smoother_update(
            prior_ens,
            posterior_ens,
            "id",
            ert.getLocalConfig(),
            ert.analysisConfig(),
            ert.rng(),
        )

    sim_gen_kw = list(prior_ens.load_parameters("SNAKE_OIL_PARAM", 0).values.flatten())

    target_gen_kw = list(
        posterior_ens.load_parameters("SNAKE_OIL_PARAM", 0).values.flatten()
    )

    assert sim_gen_kw != target_gen_kw

    assert sim_gen_kw == pytest.approx(
        [
            0.5895781800838542,
            -2.1237762127734663,
            0.22481724600587136,
            0.7564469588868706,
            0.21572672272162152,
            -0.24082711750101563,
            -1.1445220433012324,
            -1.03467093177391,
            -0.17607955213742074,
            0.02826184434039854,
        ]
    )

    assert target_gen_kw == pytest.approx(expected_gen_kw)


@pytest.mark.integration_test
def test_that_posterior_has_lower_variance_than_prior(copy_case):
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


@pytest.mark.parametrize(
    "expected_target_gen_kw, update_step",
    [
        (
            [
                0.5895781800838542,
                -1.6225405348397028,
                -0.24931876604132294,
                0.7564469588868706,
                0.21572672272162152,
                -0.24082711750101563,
                -1.1445220433012324,
                -1.03467093177391,
                -0.17607955213742074,
                0.02826184434039854,
            ],
            [
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_72"],
                    "parameters": [("SNAKE_OIL_PARAM", [1, 2])],
                }
            ],
        ),
        (
            [
                -0.6692568481556169,
                -1.6225405348397028,
                -0.22247423865074156,
                0.7564469588868706,
                0.21572672272162152,
                -0.24082711750101563,
                -1.1445220433012324,
                -1.03467093177391,
                -0.17607955213742074,
                0.02826184434039854,
            ],
            [
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_72"],
                    "parameters": [("SNAKE_OIL_PARAM", [1, 2])],
                },
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_108"],
                    "parameters": [("SNAKE_OIL_PARAM", [0, 2])],
                },
            ],
        ),
    ],
)
def test_localization(
    snake_oil_case_storage,
    snake_oil_storage,
    expected_target_gen_kw,
    update_step,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage

    ert.update_configuration = update_step

    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    prior = ert.ensemble_context(
        prior_ens,
        [True] * ert.getEnsembleSize(),
        iteration=0,
    )
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert.getEnsembleSize(),
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        prior.run_id,
        ert.getLocalConfig(),
        ert.analysisConfig(),
        ert.rng(),
    )

    sim_gen_kw = list(
        prior.sim_fs.load_parameters("SNAKE_OIL_PARAM", 0).values.flatten()
    )

    target_gen_kw = list(
        posterior_ens.load_parameters("SNAKE_OIL_PARAM", 0).values.flatten()
    )

    # Test that the localized values has been updated
    assert sim_gen_kw[1:3] != target_gen_kw[1:3]

    # test that all the other values are left unchanged
    assert sim_gen_kw[3:] == target_gen_kw[3:]

    assert target_gen_kw == pytest.approx(expected_target_gen_kw)


@pytest.mark.parametrize(
    "alpha, expected",
    [
        pytest.param(
            0.001,
            [],
            id="Low alpha, no active observations",
            marks=pytest.mark.xfail(raises=ErtAnalysisError, strict=True),
        ),
        (0.1, ["DEACTIVATED", "DEACTIVATED", "ACTIVE"]),
        (0.5, ["DEACTIVATED", "ACTIVE", "ACTIVE"]),
        (1, ["ACTIVE", "ACTIVE", "ACTIVE"]),
    ],
)
def test_snapshot_alpha(alpha, expected, storage):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """

    conf = GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )
    resp = GenDataConfig(name="RESPONSE")
    obs = xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0, 1.0, 1.0]]),
            "std": (["report_step", "index"], [[0.1, 1.0, 10.0]]),
        },
        coords={"index": [0, 1, 2], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )
    experiment = storage.create_experiment(
        parameters=[conf],
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
        prior.state_map[iens] = RealizationState.HAS_DATA
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
    w_container = SIES(prior.ensemble_size)
    analysis_config = AnalysisConfig(alpha=alpha)
    update_config = UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=["PARAMETER"],
            )
        ]
    )
    result_snapshot = iterative_smoother_update(
        prior, posterior_ens, w_container, "id", update_config, analysis_config
    )
    assert result_snapshot.alpha == alpha
    assert (
        list(result_snapshot.update_step_snapshots["ALL_ACTIVE"].obs_status) == expected
    )


@pytest.mark.integration_test
def test_that_surfaces_retain_their_order_when_loaded_and_saved_by_ert(copy_case):
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
    surf_prior = ens_prior.load_parameters("TOP", list(range(ensemble_size)))
    for i in range(ensemble_size):
        _prior_init = xtgeo.surface_from_file(
            f"surface/surf_init_{i}.irap", fformat="irap_ascii", dtype=np.float32
        )
        np.testing.assert_array_equal(surf_prior[i], _prior_init.values.data)

    surf_posterior = ens_posterior.load_parameters("TOP", list(range(ensemble_size)))

    assert surf_prior.shape == surf_posterior.shape

    for i in range(ensemble_size):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(surf_prior[i], surf_posterior[i])
        np.testing.assert_almost_equal(
            surf_prior[i].values, surf_posterior[i].values, decimal=3
        )


@pytest.mark.integration_test
def test_update_multiple_param(copy_case):
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

    sim_fs.load_parameters("SNAKE_OIL_PARAM_BPR")
    param_groups = list(sim_fs.experiment.parameter_configuration.keys())
    prior = _load_parameters(sim_fs, list(range(10)), param_groups)
    posterior = _load_parameters(posterior_fs, list(range(10)), param_groups)

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    # https://en.wikipedia.org/wiki/Variance#For_vector-valued_random_variables
    for prior_name, prior_data in prior.items():
        assert np.trace(np.cov(posterior[prior_name])) < np.trace(np.cov(prior_data))


@pytest.mark.integration_test
def test_gen_data_obs_data_mismatch(storage):
    conf = GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )
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
        parameters=[conf],
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
        prior.state_map[iens] = RealizationState.HAS_DATA
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
    analysis_config = AnalysisConfig()
    update_config = UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=["PARAMETER"],
            )
        ]
    )
    with pytest.raises(
        ErtAnalysisError,
        match="No active observations",
    ):
        smoother_update(prior, posterior_ens, "id", update_config, analysis_config)


def test_update_only_using_subset_observations(
    snake_oil_case_storage, snake_oil_storage, snapshot
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    ert.update_configuration = [
        {
            "name": "DISABLED_OBSERVATIONS",
            "observations": [
                {"name": "FOPR", "index_list": [1]},
                {"name": "WPR_DIFF_1"},
            ],
            "parameters": ert._parameter_keys,
        }
    ]
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert.getEnsembleSize(),
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        "id",
        ert.getLocalConfig(),
        ert.analysisConfig(),
    )
    log_file = Path(ert.analysisConfig().log_path) / "id.txt"
    remove_timestamp_from_logfile(log_file)
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")
