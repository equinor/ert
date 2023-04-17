from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
from iterative_ensemble_smoother import SIES

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.analysis import ErtAnalysisError, ESUpdate
from ert.analysis._es_update import _create_temporary_parameter_storage
from ert.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import open_storage


@pytest.fixture()
def minimal_config(use_tmpdir):
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write("NUM_REALIZATIONS 1")
    ert_config = ErtConfig.from_file("config_file.ert")
    yield ert_config


def test_update_report(
    snake_oil_case_storage, snake_oil_storage, new_ensemble, snapshot
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    es_update.smootherUpdate(
        snake_oil_storage.get_ensemble_by_name("default_0"),
        new_ensemble,
        "id",
    )
    log_file = Path(ert.analysisConfig().get_log_path()) / "deprecated"
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
    snake_oil_default_storage,
    module,
    expected_gen_kw,
    new_ensemble,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    ert.analysisConfig().select_module(module)
    sim_fs = snake_oil_default_storage

    if module == "IES_ENKF":
        w_container = SIES(ert.getEnsembleSize())
        es_update.iterative_smoother_update(sim_fs, new_ensemble, w_container, "id")
    else:
        es_update.smootherUpdate(sim_fs, new_ensemble, "id")

    sim_gen_kw = list(sim_fs.load_gen_kw("SNAKE_OIL_PARAM", [0]).flatten())

    target_gen_kw = list(new_ensemble.load_gen_kw("SNAKE_OIL_PARAM", [0]).flatten())

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
            "--port-range",
            "1024-65535",
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
    new_ensemble,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    # perform localization

    ert.update_configuration = update_step

    prior = ert.ensemble_context(
        snake_oil_storage.get_ensemble_by_name("default_0"),
        [True] * ert.getEnsembleSize(),
        iteration=0,
    )
    posterior = ert.ensemble_context(
        new_ensemble,
        [True] * ert.getEnsembleSize(),
        iteration=1,
    )
    es_update.smootherUpdate(prior.sim_fs, posterior.sim_fs, prior.run_id)

    sim_gen_kw = list(prior.sim_fs.load_gen_kw("SNAKE_OIL_PARAM", [0]).flatten())

    target_gen_kw = list(posterior.sim_fs.load_gen_kw("SNAKE_OIL_PARAM", [0]).flatten())

    # Test that the localized values has been updated
    assert sim_gen_kw[1:3] != target_gen_kw[1:3]

    # test that all the other values are left unchanged
    assert sim_gen_kw[3:] == target_gen_kw[3:]

    assert target_gen_kw == pytest.approx(expected_target_gen_kw)


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
@pytest.mark.parametrize(
    "alpha, expected",
    [
        pytest.param(
            0.1,
            [],
            id="Low alpha, no active observations",
            marks=pytest.mark.xfail(raises=ErtAnalysisError),
        ),
        (1, ["ACTIVE", "DEACTIVATED", "DEACTIVATED"]),
        (2, ["ACTIVE", "DEACTIVATED", "DEACTIVATED"]),
        (3, ["ACTIVE", "DEACTIVATED", "DEACTIVATED"]),
        (10, ["ACTIVE", "ACTIVE", "DEACTIVATED"]),
        (100, ["ACTIVE", "ACTIVE", "ACTIVE"]),
    ],
)
def test_snapshot_alpha(alpha, expected, snake_oil_storage, new_ensemble):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = ErtConfig.from_file("snake_oil.ert")

    obs_file = Path("observations") / "observations.txt"
    with obs_file.open(mode="w", encoding="utf-8") as fin:
        fin.write(
            """
SUMMARY_OBSERVATION LOW_STD
{
   VALUE   = 10;
   ERROR   = 0.1;
   DATE    = 2015-06-23;
   KEY     = FOPR;
};
SUMMARY_OBSERVATION HIGH_STD
{
   VALUE   = 10;
   ERROR   = 1.0;
   DATE    = 2015-06-23;
   KEY     = FOPR;
};
SUMMARY_OBSERVATION EXTREMELY_HIGH_STD
{
   VALUE   = 10;
   ERROR   = 10.0;
   DATE    = 2015-06-23;
   KEY     = FOPR;
};
"""
        )

    ert = EnKFMain(ert_config)
    es_update = ESUpdate(ert)
    ert.analysisConfig().select_module("IES_ENKF")
    sim_fs = snake_oil_storage.get_ensemble_by_name("default_0")
    ert.analysisConfig().set_enkf_alpha(alpha)
    w_container = SIES(ert.getEnsembleSize())
    es_update.iterative_smoother_update(sim_fs, new_ensemble, w_container, "id")
    result_snapshot = es_update.update_snapshots["id"]
    assert result_snapshot.alpha == alpha
    assert result_snapshot.update_step_snapshots["ALL_ACTIVE"].obs_name == [
        o
        for i, o in enumerate(
            [
                "EXTREMELY_HIGH_STD",
                "HIGH_STD",
                "LOW_STD",
            ]
        )
        if expected[i] == "ACTIVE"
    ]
    assert result_snapshot.update_step_snapshots["ALL_ACTIVE"].obs_status == expected


@pytest.mark.integration_test
def test_update_multiple_param(copy_case, new_ensemble):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    copy_case("snake_oil_field")
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "snake_oil.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)

    ert_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(ert_config)
    es_update = ESUpdate(ert)

    storage = open_storage(ert_config.ens_path)
    sim_fs = storage.get_ensemble_by_name("default")

    es_update.smootherUpdate(sim_fs, new_ensemble, "an id")

    prior = _create_temporary_parameter_storage(
        sim_fs, ert.ensembleConfig(), list(range(10))
    )
    posterior = _create_temporary_parameter_storage(
        new_ensemble, ert.ensembleConfig(), list(range(10))
    )

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    # https://en.wikipedia.org/wiki/Variance#For_vector-valued_random_variables
    for prior_name, prior_data in prior.items():
        assert np.trace(np.cov(posterior[prior_name])) < np.trace(np.cov(prior_data))


@pytest.mark.integration_test
def test_gen_data_obs_data_mismatch(snake_oil_case_storage):
    with open("observations/observations.txt", "r", encoding="utf-8") as file:
        obs_text = file.read()
    obs_text = obs_text.replace(
        "INDEX_LIST = 400,800,1200,1800;", "INDEX_LIST = 400,800,1200,1800,2400;"
    )
    with open("observations/observations.txt", "w", encoding="utf-8") as file:
        file.write(obs_text)
    with open("observations/wpr_diff_obs.txt", "a", encoding="utf-8") as file:
        file.write("0.0 0.05\n")
    ert_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(ert_config)
    es_update = ESUpdate(ert)

    with open_storage(ert.ert_config.ens_path, mode="w") as storage:
        sim_fs = storage.get_ensemble_by_name("default_0")
        target_fs = storage.create_ensemble(
            sim_fs.experiment_id,
            name="smooth",
            ensemble_size=ert.getEnsembleSize(),
            prior_ensemble=sim_fs,
        )

        with pytest.raises(
            ErtAnalysisError,
            match="Observation: WPR_DIFF_1 attached to response: SNAKE_OIL_WPR_DIFF",
        ):
            es_update.smootherUpdate(sim_fs, target_fs, "an id")
