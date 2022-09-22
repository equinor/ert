import os

import pytest

from ert._c_wrappers.enkf import ResConfig


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
