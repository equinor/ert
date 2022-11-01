import os

import pytest

from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import ResConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys


def touch(filename):
    with open(filename, "w") as fh:
        fh.write(" ")


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2554")
def test_res_config_simple_config_parsing(tmpdir):
    touch(tmpdir + "/rpfile")
    touch(tmpdir + "/datafile")
    os.mkdir(tmpdir + "/license")
    with open(tmpdir + "/test.ert", "w") as fh:
        fh.write(
            """
JOBNAME  Job%d
NUM_REALIZATIONS  1
RUNPATH_FILE rpfile
DATA_FILE datafile
LICENSE_PATH license
"""
        )
    assert (
        ResConfig(str(tmpdir + "/test.ert")).site_config
        == ResConfig(
            config_dict={
                "CONFIG_DIRECTORY": str(tmpdir),
                "DATA_FILE": "datafile",
                "LICENSE_PATH": "license",
                "RES_CONFIG_FILE": "test.ert",
                "RUNPATH_FILE": "rpfile",
            }
        ).site_config
    )


def test_res_config_minimal_dict_init(tmpdir):
    with tmpdir.as_cwd():
        config_dict = {ConfigKeys.NUM_REALIZATIONS: 1}
        res_config = ResConfig(config_dict=config_dict)
        assert res_config is not None


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")

    rconfig = None
    with pytest.raises(ConfigValidationError, match=r"Parsing.*failed"):
        rconfig = ResConfig(user_config_file=str(tmp_path / "test.ert"))

    assert rconfig is None


def test_bad_config_provide_error_message(tmp_path):
    rconfig = None
    with pytest.raises(ConfigValidationError, match=r"Error loading configuration.*"):
        testDict = {"GEN_KW": "a"}
        rconfig = ResConfig(config=testDict)

    assert rconfig is None
