import pytest

from ert._c_wrappers.enkf import SummaryConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_summary_config():
    summary_config = SummaryConfig(name="ALT1", input_file="ECLBASE", keys=[])
    assert summary_config.name == "ALT1"
