import pytest

from ert.config.gen_data_config import GenDataConfig
from ert.config.responses_index import responses_index
from ert.config.summary_config import SummaryConfig


def test_adding_gendata_and_summary():
    ri = responses_index

    # Manually reset it
    ri._items = {}

    assert [*ri.keys()] == []
    assert [*ri.values()] == []
    assert [*ri.items()] == []

    ri.add_response_type(GenDataConfig)
    assert [*ri.keys()] == ["GenDataConfig"]
    assert [*ri.values()] == [GenDataConfig]
    assert [*ri.items()] == [("GenDataConfig", GenDataConfig)]

    with pytest.raises(
        KeyError, match="Response type with name GenDataConfig is already registered"
    ):
        ri.add_response_type(GenDataConfig)

    ri.add_response_type(SummaryConfig)
    assert [*ri.keys()] == ["GenDataConfig", "SummaryConfig"]
    assert [*ri.values()] == [GenDataConfig, SummaryConfig]
    assert [*ri.items()] == [
        ("GenDataConfig", GenDataConfig),
        ("SummaryConfig", SummaryConfig),
    ]

    with pytest.raises(
        KeyError, match="Response type with name SummaryConfig is already registered"
    ):
        ri.add_response_type(SummaryConfig)


def test_adding_non_response_config():
    ri = responses_index

    class NotAResponseConfig:
        pass

    with pytest.raises(
        ValueError, match="Response type must be subclass of ResponseConfig"
    ):
        ri.add_response_type(NotAResponseConfig)
