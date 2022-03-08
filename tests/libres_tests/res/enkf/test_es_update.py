import sys

import pytest

from ecl.util.util import BoolVector

from res.enkf import EnkfNode, ErtRunContext, ESUpdate, NodeId, EnKFMain


@pytest.mark.parametrize(
    "module",
    [
        "IES_ENKF",
        "STD_ENKF",
    ],
)
def test_get_module(setup_case, module):
    res_config = setup_case("local/mini_ert", "mini_config")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)

    es_update.getModule(module)


@pytest.mark.parametrize(
    "module, expected", [("NO_NOT_THIS_MODULE", False), ("STD_ENKF", True)]
)
def test_has_module(setup_case, module, expected):
    res_config = setup_case("local/mini_ert", "mini_config")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)

    assert es_update.hasModule(module) is expected


def test_get_invalid_module(setup_case):
    res_config = setup_case("local/mini_ert", "mini_config")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)

    with pytest.raises(KeyError, match="No such module:STD_ENKF_XXX"):
        es_update.getModule("STD_ENKF_XXX")


@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        pytest.param(
            "IES_ENKF",
            [
                -1.1382059686811712,
                -0.10706650113784337,
                -0.5321174510859479,
                0.16314430250454545,
                -0.7140756660858336,
                -1.5863225705846462,
                -0.061532565899646424,
                -0.3881321947200636,
                1.0849199162367573,
                -0.2026982960301163,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("darwin"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "IES_ENKF",
            [
                -1.1286562142358219,
                -0.07357477296944821,
                -0.5840952488968133,
                0.3057016147126608,
                -0.6721257967032032,
                -1.545169986269137,
                -0.4528695575362665,
                -0.3521622538356382,
                1.0495218946126155,
                -0.23061011158425496,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("linux"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "STD_ENKF",
            [
                -1.0279886752792073,
                -0.7266247822582957,
                -0.12686045273301966,
                -0.22659509892534876,
                -1.120792349644604,
                -1.4956547646687188,
                -0.15332787834835337,
                -0.9161628056286735,
                1.7415219166473932,
                -0.4014927497171072,
            ],
            marks=pytest.mark.skipif(
                sys.platform.startswith("darwin"),
                reason="See https://github.com/equinor/ert/issues/2351",
            ),
        ),
        pytest.param(
            "STD_ENKF",
            [
                -1.0120724178702936,
                -0.6708052353109712,
                -0.2134901157511208,
                0.01100042142150413,
                -1.0508759006735606,
                -1.4270671241428736,
                -0.8055561977427175,
                -0.8562129041546276,
                1.682525213940497,
                -0.4480124423073352,
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
    run_context = ErtRunContext.ensemble_smoother_update(sim_fs, target_fs)
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
                0.6759029719309165,
                -0.7802509588954853,
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
                0.867127147787121,
                -1.2502532947839953,
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
    localized_idxs = (1, 2)
    local_config = ert.getLocalConfig()
    local_config.clear()
    obs = local_config.createObsdata("OBSSET_LOCA")
    obs.addNode("WOPR_OP1_72")
    ministep = local_config.createMinistep("MINISTEP_LOCA")
    ministep.addActiveData("SNAKE_OIL_PARAM")  # replace dataset.addNode()
    ministep.activate_indices("SNAKE_OIL_PARAM", localized_idxs)
    ministep.attachObsset(obs)
    updatestep = local_config.getUpdatestep()
    updatestep.attachMinistep(ministep)

    # Run enseble smoother
    mask = BoolVector(initial_size=ert.getEnsembleSize(), default_value=True)
    model_config = ert.getModelConfig()
    path_fmt = model_config.getRunpathFormat()
    jobname_fmt = model_config.getJobnameFormat()
    subst_list = None
    run_context = ErtRunContext.ensemble_smoother(
        sim_fs, target_fs, mask, path_fmt, jobname_fmt, subst_list, 0
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
