import pytest
import numpy as np

from ert._c_wrappers.enkf.data.enkf_node import EnkfNode
from ert._c_wrappers.enkf.node_id import NodeId


def test_load_active_masks(snake_oil_case_storage):
    case1 = "default_0"
    ert = snake_oil_case_storage

    array_size = 15  # arbitrary size
    report_step = 199
    realization = 0
    node_id = NodeId(report_step, realization)

    fs1 = ert.getEnkfFsManager().getFileSystem(case1)
    config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")
    fs1.save_gen_data(
        config_node.getKey(), np.ndarray(array_size), report_step, realization
    )

    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs1, node_id)

    active_mask = config_node.getDataModelConfig().getActiveMask()
    assert len(active_mask) == array_size

    active_mask = config_node.getDataModelConfig().getActiveMask()
    assert len(active_mask) == array_size

    # Setting one element to False, load different case, check, reload,
    # and check.
    assert active_mask[10]
    active_mask_modified = active_mask.copy()
    active_mask_modified[10] = False

    # Load first - check element is true
    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs1, node_id)
    active_mask = config_node.getDataModelConfig().getActiveMask()
    assert active_mask[10]


def test_create(snake_oil_case_storage):
    ert = snake_oil_case_storage
    fs1 = ert.getEnkfFsManager().getCurrentFileSystem()
    config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")

    fs1.save_gen_data(
        config_node.getKey(), np.ndarray(array_size), report_step, realization
    )

    data_node = EnkfNode(config_node)
    assert data_node.tryLoad(fs1, NodeId(report_step, realization))

    gen_data = data_node.asGenData()
    data = gen_data.getData()

    assert len(data) == array_size
