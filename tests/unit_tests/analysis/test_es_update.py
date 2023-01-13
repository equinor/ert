from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
from iterative_ensemble_smoother import IterativeEnsembleSmoother

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, EnkfNode, ErtConfig, NodeId
from ert.analysis import ErtAnalysisError, ESUpdate
from ert.analysis._es_update import _create_temporary_parameter_storage
from ert.cli import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli


def test_update_report(snake_oil_case_storage, snapshot):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    fsm = ert.storage_manager
    es_update.smootherUpdate(fsm["default_0"], fsm.add_case("target"), "id")
    log_file = Path(ert.analysisConfig().get_log_path()) / "deprecated"
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")


@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        (
            "IES_ENKF",
            [
                0.1964238785703465,
                -0.7864807076941146,
                -0.5807896159251078,
                -0.22435575683095582,
                -0.2021017211719207,
                0.07367941192265133,
                -1.4227423506253276,
                0.2571096893111238,
                -0.4974944985473074,
                0.7081334216380905,
            ],
        ),
        (
            "STD_ENKF",
            [
                -0.06567898910532588,
                0.10504962902545388,
                -1.1178608572124271,
                -0.8782242339761734,
                -0.4806540171009489,
                0.28335043153842926,
                -1.6082225555080574,
                1.1182967700344797,
                -0.7117711294872316,
                1.161381139836552,
            ],
        ),
    ],
)
def test_update_snapshot(snake_oil_case_storage, module, expected_gen_kw):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    ert.analysisConfig().select_module(module)
    fsm = ert.storage_manager
    sim_fs = fsm["default_0"]
    target_fs = fsm.add_case("target")

    if module == "IES_ENKF":
        w_container = IterativeEnsembleSmoother(ert.getEnsembleSize())
        es_update.iterative_smoother_update(sim_fs, target_fs, w_container, "id")
    else:
        es_update.smootherUpdate(sim_fs, target_fs, "id")

    conf = ert.ensembleConfig()["SNAKE_OIL_PARAM"]
    sim_node = EnkfNode(conf)
    target_node = EnkfNode(conf)

    node_id = NodeId(0, 0)
    sim_node.load(sim_fs, node_id)
    target_node.load(target_fs, node_id)

    sim_gen_kw = list(sim_node.asGenKw())
    target_gen_kw = list(target_node.asGenKw())

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
    df_default = facade.load_all_gen_kw_data("default")
    df_target = facade.load_all_gen_kw_data("target")

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
                -0.732401196722254,
                -1.0913320833716744,
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
                -1.5836182533774308,
                -0.732401196722254,
                -0.8951194275529923,
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
def test_localization(snake_oil_case_storage, expected_target_gen_kw, update_step):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert = snake_oil_case_storage
    es_update = ESUpdate(ert)
    # perform localization

    ert.update_configuration = update_step

    prior = ert.load_ensemble_context(
        "default_0", [True] * ert.getEnsembleSize(), iteration=0
    )
    posterior = ert.create_ensemble_context(
        "target", [True] * ert.getEnsembleSize(), iteration=1
    )
    es_update.smootherUpdate(prior.sim_fs, posterior.sim_fs, prior.run_id)

    conf = ert.ensembleConfig()["SNAKE_OIL_PARAM"]
    sim_node = EnkfNode(conf)
    target_node = EnkfNode(conf)

    node_id = NodeId(0, 0)
    sim_node.load(prior.sim_fs, node_id)
    target_node.load(posterior.sim_fs, node_id)

    sim_gen_kw = list(sim_node.asGenKw())
    target_gen_kw = list(target_node.asGenKw())

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
        (10, ["ACTIVE", "DEACTIVATED", "ACTIVE"]),
        (100, ["ACTIVE", "ACTIVE", "ACTIVE"]),
    ],
)
def test_snapshot_alpha(alpha, expected):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    res_config = ErtConfig.from_file("snake_oil.ert")

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

    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    ert.analysisConfig().select_module("IES_ENKF")
    fsm = ert.storage_manager
    sim_fs = fsm["default_0"]
    target_fs = fsm.add_case("target")
    ert.analysisConfig().set_enkf_alpha(alpha)
    w_container = IterativeEnsembleSmoother(ert.getEnsembleSize())
    es_update.iterative_smoother_update(sim_fs, target_fs, w_container, "id")
    result_snapshot = es_update.update_snapshots["id"]
    assert result_snapshot.alpha == alpha
    assert result_snapshot.update_step_snapshots["ALL_ACTIVE"].obs_status == expected


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
            ENSEMBLE_EXPERIMENT_MODE,
            "snake_oil.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)

    res_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    fsm = ert.storage_manager
    sim_fs = fsm["default"]
    target_fs = fsm.add_case("target")

    es_update.smootherUpdate(sim_fs, target_fs, "an id")

    prior = _create_temporary_parameter_storage(
        sim_fs, ert.ensembleConfig(), list(range(10))
    )
    posterior = _create_temporary_parameter_storage(
        target_fs, ert.ensembleConfig(), list(range(10))
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
    res_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    fsm = ert.storage_manager
    sim_fs = fsm["default_0"]
    target_fs = fsm.add_case("smooth")
    with pytest.raises(
        ErtAnalysisError,
        match="Observation: WPR_DIFF_1 attached to response: SNAKE_OIL_WPR_DIFF",
    ):
        es_update.smootherUpdate(sim_fs, target_fs, "an id")
