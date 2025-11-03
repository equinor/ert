import pytest
from hypothesis import given
from resfo_utilities.testing import summaries

from ert.config._read_summary import read_summary


@pytest.mark.timeout(100)
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
    fetch_keys = [str(i) for i in range(1000)]
    (_, keys, time_map, _) = read_summary(str(tmp_path / "TEST"), fetch_keys)
    assert all(k in fetch_keys for k in keys)
    assert len(time_map) == len(unsmry.steps)
