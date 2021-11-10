import os
import shutil

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


def test_update(setup_case):
    res_config = setup_case("local/snake_oil", "snake_oil.ert")

    ert = EnKFMain(res_config)
    es_update = ESUpdate(ert)
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

    sim_gen_kw = sim_node.asGenKw()
    target_gen_kw = target_node.asGenKw()

    assert list(sim_gen_kw) != list(target_gen_kw)


def test_localization(setup_case):
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
