import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import AnalysisConfig, EclConfig, EnkfConfigNode
from ert._c_wrappers.enkf.util import TimeMap


def test_failed_class_creation():
    classes = [EnkfConfigNode]

    for cls in classes:
        with pytest.raises(NotImplementedError):
            cls()


@pytest.mark.unstable
def test_ecl_config_creation(minimum_example):
    assert isinstance(minimum_example.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_example.eclConfig(), EclConfig)

    with pytest.raises(AssertionError):  # Null pointer!
        assert isinstance(minimum_example.eclConfig().getRefcase(), EclSum)

    file_system = minimum_example.getEnkfFsManager().getCurrentFileSystem()
    assert file_system.getCaseName() == "default"
    time_map = file_system.getTimeMap()
    assert isinstance(time_map, TimeMap)
