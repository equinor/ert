from ert._c_wrappers.enkf import SummaryConfig


def test_summary_config():
    summary_config = SummaryConfig(name="ALT1", input_file="ECLBASE", keys=[])
    assert summary_config.name == "ALT1"


def test_summary_config_equal():
    summary_config = SummaryConfig(name="n", input_file="file", keys=[])

    new_summary_config = SummaryConfig(name="n", input_file="file", keys=[])
    assert summary_config == new_summary_config

    new_summary_config = SummaryConfig(name="different", input_file="file", keys=[])
    assert summary_config != new_summary_config

    new_summary_config = SummaryConfig(name="n", input_file="different", keys=[])
    assert summary_config != new_summary_config

    new_summary_config = SummaryConfig(name="n", input_file="file", keys=["some_key"])
    assert summary_config != new_summary_config
