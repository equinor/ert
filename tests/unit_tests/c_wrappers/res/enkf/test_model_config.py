import pytest

from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf import ResConfig


def test_eclbase_and_jobname():
    """given eclbase and jobname, the jobname format string should not be
    overridden by eclbase"""
    res_config = ResConfig(
        config_dict={
            "NUM_REALIZATIONS": 1,
            "ENSPATH": "Ensemble",
            "ECLBASE": "ECLBASE%d",
            "JOBNAME": "JOBNAME%d",
        }
    )
    assert res_config.model_config.jobname_format_string == "JOBNAME%d"


def test_eclbase_gets_parsed_as_jobname_format_string_when_jobname_not_set():
    res_config = ResConfig(
        config_dict={
            "NUM_REALIZATIONS": 1,
            "ENSPATH": "Ensemble",
            "ECLBASE": "ECLBASE%d",
        }
    )
    assert res_config.model_config.jobname_format_string == "ECLBASE%d"


def test_that_summary_given_without_eclbase_gives_error(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nSUMMARY summary")
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="When using SUMMARY keyword, the config must also specify ECLBASE",
    ):
        ResConfig(user_config_file=str(tmp_path / "config.ert"))


def test_jobname_gets_parsed_to_jobname_format_string():
    res_config = ResConfig(
        config_dict={
            "NUM_REALIZATIONS": 1,
            "ENSPATH": "Ensemble",
            "JOBNAME": "JOBNAME%d",
        }
    )
    assert res_config.model_config.jobname_format_string == "JOBNAME%d"
