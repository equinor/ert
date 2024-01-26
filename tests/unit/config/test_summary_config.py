import pytest

from ert.config import ConfigValidationError, ErtConfig, SummaryConfig


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


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REALIZATIONS 10\n SUMMARY FOPR")
    with pytest.raises(
        ConfigValidationError,
        match=r"Line 2 .* When using SUMMARY keyword,"
        " the config must also specify ECLBASE",
    ):
        _ = ErtConfig.from_file(str(tmp_path / "test.ert"))
