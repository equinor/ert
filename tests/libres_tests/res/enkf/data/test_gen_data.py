from ert._c_wrappers.enkf.data.enkf_node import EnkfNode
from ert._c_wrappers.enkf.node_id import NodeId


def test_create(snake_oil_case):
    ert = snake_oil_case
    fs1 = ert.getEnkfFsManager().getCurrentFileSystem()
    config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")

    data_node = EnkfNode(config_node)
    data_node.tryLoad(fs1, NodeId(199, 0))

    gen_data = data_node.asGenData()
    data = gen_data.getData()

    assert len(data) == 2000
