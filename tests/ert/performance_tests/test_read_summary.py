from hypothesis import given

from ert.config._read_summary import read_summary
from tests.ert.unit_tests.config.summary_generator import (
    summaries,
)


@given(summaries())
def test_that_length_of_fetch_keys_does_not_reduce_performance(
    tmp_path_factory, summary
):
    """With a compiled regex this takes seconds to run, and with
    a naive implementation it will take almost an hour.
    """
    tmp_path = tmp_path_factory.mktemp("summary")
    smspec, unsmry = summary
    unsmry.to_file(tmp_path / "TEST.UNSMRY")
    smspec.to_file(tmp_path / "TEST.SMSPEC")
    fetch_keys = [str(i) for i in range(100000)]
    (_, keys, time_map, _) = read_summary(str(tmp_path / "TEST"), fetch_keys)
    assert all(k in fetch_keys for k in keys)
    assert len(time_map) == len(unsmry.steps)
