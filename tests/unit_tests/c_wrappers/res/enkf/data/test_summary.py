import pytest

from ert._c_wrappers.enkf.config import SummaryConfig
from ert._c_wrappers.enkf.data.summary import Summary


def test_create():
    config = SummaryConfig("WWCT:OP_5")
    summary = Summary(config)
    assert len(summary) == 0

    with pytest.raises(IndexError):
        _ = summary[100]

    summary[0] = 75
    assert summary[0] == 75

    summary[10] = 100
    assert summary[10] == 100

    with pytest.raises(ValueError):
        _ = summary[5]
