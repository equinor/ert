from pathlib import Path

import sys

import pytest

from ert.analysis import ESUpdate, ErtAnalysisError
from res.enkf import (
    EnkfNode,
    ErtRunContext,
    NodeId,
    EnKFMain,
    ResConfig,
)
from res._lib import ies


@pytest.fixture()
def minimal_config(use_tmpdir):
    with open("config_file.ert", "w") as fout:
        fout.write("NUM_REALIZATIONS 1")
    res_config = ResConfig("config_file.ert")
    yield res_config


def test_update_report(setup_case, snapshot):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    res_config = setup_case("local/snake_oil", "snake_oil.ert")

    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    fsm = ert.getEnkfFsManager()
    sim_fs = fsm.getFileSystem("default_0")
    target_fs = fsm.getFileSystem("target")
    run_context = ErtRunContext(sim_fs, target_fs)
    es_update.smootherUpdate(run_context)
    log_file = Path(ert.analysisConfig().get_log_path()) / "deprecated"
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")


@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        pytest.param(
            "IES_ENKF",
            [
                0.19646461221617806,
                -0.8580650655205431,
                -1.7033885637717783,
                0.2143913913285077,
                -0.6416317135179632,
                -0.9655917323739145,
                0.33111809209873133,
                -0.39486155020884783,
                1.6583030240523025,
                -1.2083505536978683,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("darwin"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "IES_ENKF",
            [
                0.4017777704668615,
                -0.9277413075241868,
                -1.8045946565215083,
                0.23176385196876714,
                -0.8300453517964581,
                -0.9828660162918067,
                -0.051783268233953655,
                -0.40210522896061107,
                1.8186815661449791,
                -1.1306332986095615,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("linux"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "STD_ENKF",
            [
                1.196462292883038,
                -1.9782890562294615,
                -2.078978973876065,
                -0.14118328421874526,
                -1.0000524286981523,
                -0.461103367650835,
                0.5010898849822789,
                -0.9273783981099771,
                2.6971604296733,
                -2.077579845830024,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("darwin"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "STD_ENKF",
            [
                1.5386508899675102,
                -2.0944161262355347,
                -2.247655795125614,
                -0.11222918315164569,
                -1.3140751591623103,
                -0.4898938408473217,
                -0.13707904890552972,
                -0.9394511960295817,
                2.9644579998277614,
                -1.948051087349513,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("linux"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
    ],
)
def test_update(setup_case, module, expected_gen_kw):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    res_config = setup_case("local/snake_oil", "snake_oil.ert")

    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    ert.analysisConfig().selectModule(module)
    fsm = ert.getEnkfFsManager()
    sim_fs = fsm.getFileSystem("default_0")
    target_fs = fsm.getFileSystem("target")
    run_context = ErtRunContext(sim_fs, target_fs)

    if module == "IES_ENKF":
        w_container = ies.ModuleData(ert.getEnsembleSize())
        es_update.iterative_smoother_update(run_context, w_container)
    else:
        es_update.smootherUpdate(run_context)

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
            -1.3035319087841115,
            0.8222709205428339,
            -1.1400029486153482,
            0.7477534046493867,
            -0.10400064074767973,
            -1.7223242794585338,
            0.0761604027734105,
            0.4039137216428462,
            0.10001691562080614,
            0.09549338450036506,
        ]
    )

    assert target_gen_kw == pytest.approx(expected_gen_kw)


@pytest.mark.parametrize(
    "expected_target_gen_kw",
    [
        pytest.param(
            [
                -1.3035319087841115,
                0.478900928107711,
                -0.2960474299371837,
                0.7477534046493867,
                -0.10400064074767973,
                -1.7223242794585338,
                0.0761604027734105,
                0.4039137216428462,
                0.10001691562080614,
                0.09549338450036506,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("darwin"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            [
                -1.3035319087841115,
                0.927500805607199,
                -1.3986433196044348,
                0.7477534046493867,
                -0.10400064074767973,
                -1.7223242794585338,
                0.0761604027734105,
                0.4039137216428462,
                0.10001691562080614,
                0.09549338450036506,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("linux"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
    ],
)
def test_localization(setup_case, expected_target_gen_kw):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    res_config = setup_case("local/snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
    fsm = ert.getEnkfFsManager()
    sim_fs = fsm.getFileSystem("default_0")
    target_fs = fsm.getFileSystem("target")
    # perform localization
    update_step = [
        {
            "name": "update_step_LOCA",
            "observations": ["WOPR_OP1_72"],
            "parameters": [("SNAKE_OIL_PARAM", [1, 2])],
        }
    ]

    ert.update_configuration = update_step

    run_context = ert.create_ensemble_smoother_run_context(
        source_filesystem=sim_fs, target_filesystem=target_fs, iteration=0
    )
    es_update.smootherUpdate(run_context)

    conf = ert.ensembleConfig()["SNAKE_OIL_PARAM"]
    sim_node = EnkfNode(conf)
    target_node = EnkfNode(conf)

    node_id = NodeId(0, 0)
    sim_node.load(sim_fs, node_id)
    target_node.load(target_fs, node_id)

    sim_gen_kw = list(sim_node.asGenKw())
    target_gen_kw = list(target_node.asGenKw())

    # Test that the localized values has been updated
    assert sim_gen_kw[1:3] != target_gen_kw[1:3]

    # test that all the other values are left unchanged
    assert sim_gen_kw[3:] == target_gen_kw[3:]
    assert sim_gen_kw[0] == target_gen_kw[0]

    assert target_gen_kw == pytest.approx(expected_target_gen_kw)


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
def test_snapshot_alpha(setup_case, alpha, expected):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    res_config = setup_case("local/snake_oil", "snake_oil.ert")

    obs_file = Path("observations") / "observations.txt"
    with obs_file.open(mode="w") as fin:
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
    ert.analysisConfig().selectModule("IES_ENKF")
    fsm = ert.getEnkfFsManager()
    sim_fs = fsm.getFileSystem("default_0")
    target_fs = fsm.getFileSystem("target")
    run_context = ErtRunContext(sim_fs, target_fs)
    ert.analysisConfig().setEnkfAlpha(alpha)
    w_container = ies.ModuleData(ert.getEnsembleSize())
    w_container.iteration_nr += 1
    es_update.iterative_smoother_update(run_context, w_container)
    result_snapshot = es_update.update_snapshots[run_context.get_id()]
    assert result_snapshot.alpha == alpha
    assert result_snapshot.update_step_snapshots["ALL_ACTIVE"].obs_status == expected
