from ert._c_wrappers.enkf import NodeId
from ert._c_wrappers.enkf.config import GenDataConfig
from ert._c_wrappers.enkf.data import EnkfNode


def load_active_masks(snake_oil_case):
    case1 = "default_0"
    case2 = "default_1"
    ert = snake_oil_case

    fs1 = ert.getEnkfFsManager().getFileSystem(case1)
    config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")
    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs1, NodeId(199, 0))

    active_mask = config_node.getDataModelConfig().getActiveMask()
    first_active_mask_length = len(active_mask)
    assert first_active_mask_length == 2000

    fs2 = ert.getEnkfFsManager().getFileSystem(case2)
    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs2, NodeId(199, 0))

    active_mask = config_node.getDataModelConfig().getActiveMask()
    second_active_mask_len = len(active_mask)
    assert second_active_mask_len == 2000
    assert first_active_mask_length == second_active_mask_len

    # Setting one element to False, load different case, check, reload,
    # and check.
    assert active_mask[10]
    active_mask_modified = active_mask.copy()
    active_mask_modified[10] = False

    # Load first - check element is true
    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs1, NodeId(199, 0))
    active_mask = config_node.getDataModelConfig().getActiveMask()
    assert active_mask[10]


def test_create():
    GenDataConfig("KEY")
