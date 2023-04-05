import pytest

from ert.parsing import ConfigValidationError
from ert._c_wrappers.enkf import ErtConfig


@pytest.mark.parametrize(
    "extra_config, expected",
    [
        pytest.param(
            {"ECLBASE": "ECLBASE%d", "JOBNAME": "JOBNAME%d"},
            "JOBNAME<IENS>",
            id="ECLBASE does not overwrite JOBNAME",
        ),
        pytest.param(
            {"ECLBASE": "ECLBASE%d"},
            "ECLBASE<IENS>",
            id="ECLBASE is also used as JOBNAME",
        ),
        pytest.param(
            {"ECLBASE": "ECLBASE%d", "JOBNAME": "JOBNAME%d"},
            "JOBNAME<IENS>",
            id="JOBNAME is used when no ECLBASE is present",
        ),
    ],
)
def test_model_config_jobname_and_eclbase(extra_config, expected):
    config_dict = {"NUM_REALIZATIONS": 1, "ENSPATH": "Ensemble", **extra_config}
    ert_config = ErtConfig.from_dict(config_dict)
    assert ert_config.model_config.jobname_format_string == expected


def test_that_summary_given_without_eclbase_gives_error(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nSUMMARY summary")
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="When using SUMMARY keyword, the config must also specify ECLBASE",
    ):
        ErtConfig.from_file(str(tmp_path / "config.ert"))
