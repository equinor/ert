import pytest

from ert.shared.status.utils import format_running_time


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "Running time: 0 seconds"),
        (1, "Running time: 1 seconds"),
        (100, "Running time: 1 minutes 40 seconds"),
        (10000, "Running time: 2 hours 46 minutes 40 seconds"),
        (100000, "Running time: 1 days 3 hours 46 minutes 40 seconds"),
        (100000000, "Running time: 1157 days 9 hours 46 minutes 40 seconds"),
    ],
)
def test_format_running_time(seconds, expected):
    assert format_running_time(seconds) == expected
