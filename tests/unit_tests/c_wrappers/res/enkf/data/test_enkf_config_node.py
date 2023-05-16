import pytest

from ert._c_wrappers.enkf.config import EnkfConfigNode


def test_failed_enkf_config_node_creation():
    with pytest.raises(NotImplementedError):
        EnkfConfigNode()
