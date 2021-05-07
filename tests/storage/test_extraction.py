import pytest
import pandas as pd
from ert_shared.storage.extraction import _prepare_x_axis


@pytest.mark.parametrize(
    "x_axis, expected",
    [
        ([1, 2, 3, 4], ["1", "2", "3", "4"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (
            [pd.Timestamp(x, unit="d") for x in range(4)],
            [
                "1970-01-01T00:00:00",
                "1970-01-02T00:00:00",
                "1970-01-03T00:00:00",
                "1970-01-04T00:00:00",
            ],
        ),
    ],
)
def test_prepare_x_axis(x_axis, expected):
    assert expected == _prepare_x_axis(x_axis)
