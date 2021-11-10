import os
import shutil
import sys

import pytest

from ecl.util.util import BoolVector

from res.enkf import EnkfNode, ErtRunContext, ESUpdate, NodeId, ResConfig, EnKFMain


@pytest.fixture()
def setup_case(tmpdir, source_root):
    def copy_case(path, config_file):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        os.chdir("test_data")
        return ResConfig(config_file)

    with tmpdir.as_cwd():
        yield copy_case


def test_create(setup_case):
    res_config = setup_case("local/mini_ert", "mini_config")
    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)

    assert not es_update.hasModule("NO_NOT_THIS_MODULE")
    with pytest.raises(KeyError):
        es_update.getModule("STD_ENKF_XXX")

    es_update.getModule("STD_ENKF")


@pytest.mark.xfail(sys.platform == "darwin", reason="Different result from update")
@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        (
            "BOOTSTRAP_ENKF",
            [
                -0.13199749242620085,
                0.19931631512450643,
                -1.4191070405813302,
                0.6834759421494758,
                0.2036333008086772,
                -1.800805191591989,
                2.087497120639354,
                0.6706824284593408,
                1.202301603428107,
                -0.2991274236216228,
            ],
        ),
        (
            "CV_ENKF",
            [
                -1.3552992690431354,
                0.8111762768114376,
                -1.041495570212803,
                0.7579452728955424,
                -0.11919680524287896,
                -1.69637652448712,
                0.03478817290978127,
                0.4178555072326893,
                0.029729592107867884,
                0.10861381054387127,
            ],
        ),
        (
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
        ),
        # "NULL_ENKF",  # This does nothing, so the test will not pass
        (
            "SQRT_ENKF",
            [
                -2.9193320576762396,
                3.1964806431964665,
                -4.602308580793938,
                2.6664788740452723,
                -7.7069598054095945,
                -2.5899923144860697,
                -1.8672280427461743,
                3.8585956848850573,
                -0.10031545644060243,
                0.6134116781275245,
            ],
        ),
        (
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
        ),
        (
            "STD_ENKF_DEBUG",
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


@pytest.mark.xfail(sys.platform == "darwin", reason="Different result from update")
def test_localization(setup_case):
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
    dataset = local_config.createDataset("DATASET_SCALAR_LOCA")
    dataset.addNode("SNAKE_OIL_PARAM")
    active_list = dataset.getActiveList("SNAKE_OIL_PARAM")
    for i in localized_idxs:
        active_list.addActiveIndex(i)
    obs = local_config.createObsdata("OBSSET_LOCA")
    obs.addNode("WOPR_OP1_72")
    ministep = local_config.createMinistep("MINISTEP_LOCA")
    ministep.attachDataset(dataset)
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

    assert target_gen_kw == pytest.approx(
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
        ]
    )
