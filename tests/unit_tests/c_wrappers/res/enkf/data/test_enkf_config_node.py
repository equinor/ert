import pytest

from ert._c_wrappers.enkf.config import EnkfConfigNode


def test_failed_enkf_config_node_creation():
    with pytest.raises(NotImplementedError):
        EnkfConfigNode()


def test_gen_data():
    assert not EnkfConfigNode.validate_gen_data_format("FILE")
    assert EnkfConfigNode.validate_gen_data_format("FILE%d")
    assert not EnkfConfigNode.validate_gen_data_format("FILE%d%d")

    config_node = EnkfConfigNode.create_gen_data("KEY")
    assert isinstance(config_node, EnkfConfigNode)
    gen_data = config_node.getModelConfig()
    assert gen_data.getNumReportStep() == 1
    assert gen_data.getReportStep(0) == 0

    config_node = EnkfConfigNode.create_gen_data("KEY", report_steps=[10, 20, 30])
    assert isinstance(config_node, EnkfConfigNode)
    gen_data = config_node.getModelConfig()
    assert gen_data.getNumReportStep() == 3
    for r1, r2 in zip([10, 20, 30], gen_data.getReportSteps()):
        assert r1 == r2
