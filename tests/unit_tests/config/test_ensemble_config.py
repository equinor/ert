import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
import xtgeo
from ecl.summary import EclSum

from ert.config import ConfigValidationError, EnsembleConfig, ErtConfig
from ert.config.parsing import ConfigKeys


def test_create():
    empty_ens_conf = EnsembleConfig()
    conf_from_dict = EnsembleConfig.from_dict({})

    assert empty_ens_conf == conf_from_dict
    assert conf_from_dict.grid_file is None
    assert not conf_from_dict.parameters

    assert "XYZ" not in conf_from_dict

    with pytest.raises(KeyError):
        _ = conf_from_dict["KEY"]


@pytest.mark.usefixtures("use_tmpdir")
def test_ensemble_config_fails_on_non_sensical_refcase_file():
    refcase_file = "CEST_PAS_UNE_REFCASE"
    refcase_file_content = """
_________________________________________     _____    ____________________
\\______   \\_   _____/\\_   _____/\\_   ___ \\   /  _  \\  /   _____/\\_   _____/
 |       _/|    __)_  |    __)  /    \\  \\/  /  /_\\  \\ \\_____  \\  |    __)_
 |    |   \\|        \\ |     \\   \\     \\____/    |    \\/        \\ |        \\
 |____|_  /_______  / \\___  /    \\______  /\\____|__  /_______  //_______  /
        \\/        \\/      \\/            \\/         \\/        \\/         \\/
"""
    with open(refcase_file + ".UNSMRY", "w+", encoding="utf-8") as refcase_file_handler:
        refcase_file_handler.write(refcase_file_content)
    with open(refcase_file + ".SMSPEC", "w+", encoding="utf-8") as refcase_file_handler:
        refcase_file_handler.write(refcase_file_content)
    with pytest.raises(expected_exception=IOError, match=refcase_file):
        config_dict = {ConfigKeys.REFCASE: refcase_file}
        EnsembleConfig.from_dict(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
def test_ensemble_config_construct_refcase_and_grid():
    grid_file = "CASE.EGRID"
    refcase_file = "REFCASE_NAME"
    xtgeo.create_box_grid(dimension=(10, 10, 1)).to_file("CASE.EGRID", "egrid")
    ecl_sum = EclSum.writer("REFCASE_NAME", datetime(2014, 9, 10), 3, 3, 3)
    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    t_step = ecl_sum.addTStep(1, sim_days=10)
    t_step["FOPR"] = 10
    ecl_sum.fwrite()
    ec = EnsembleConfig.from_dict(
        config_dict={
            ConfigKeys.GRID: grid_file,
            ConfigKeys.REFCASE: refcase_file,
        },
    )

    assert isinstance(ec, EnsembleConfig)
    assert isinstance(ec.refcase, EclSum)

    assert ec.grid_file == os.path.realpath(grid_file)


def test_that_refcase_gets_correct_name(tmpdir):
    refcase_name = "MY_REFCASE"
    config_dict = {
        ConfigKeys.REFCASE: refcase_name,
    }

    with tmpdir.as_cwd():
        ecl_sum = EclSum.writer(refcase_name, datetime(2014, 9, 10), 10, 10, 10)
        ecl_sum.addVariable("FOPR", unit="SM3/DAY")
        t_step = ecl_sum.addTStep(2, sim_days=1)
        t_step["FOPR"] = 1
        ecl_sum.fwrite()

        ec = EnsembleConfig.from_dict(config_dict=config_dict)
        assert os.path.realpath(refcase_name) == ec.refcase.case


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "existing_suffix, expected_suffix",
    [
        pytest.param(
            "UNSMRY",
            "SMSPEC",
        ),
        pytest.param(
            "SMSPEC",
            "UNSMRY",
        ),
    ],
)
def test_that_files_for_refcase_exists(existing_suffix, expected_suffix):
    refcase_file = "missing_refcase_file"

    with open(
        refcase_file + "." + existing_suffix, "w+", encoding="utf-8"
    ) as refcase_writer:
        refcase_writer.write("")

    with pytest.raises(
        ConfigValidationError,
        match="Cannot find " + expected_suffix + " file for refcase provided!",
    ):
        _ = EnsembleConfig.from_dict(
            config_dict={
                ConfigKeys.REFCASE: refcase_file,
            },
        )


def test_ensemble_config_duplicate_node_names():
    duplicate_name = "Test_name"
    Path("MULTFLT.TXT").write_text("", encoding="utf-8")
    Path("FAULT_TEMPLATE").write_text("", encoding="utf-8")
    config_dict = {
        ConfigKeys.GEN_DATA: [
            [
                duplicate_name,
                "INPUT_FORMAT:ASCII",
                "RESULT_FILE:snake_oil_opr_diff_%d.txt",
                "REPORT_STEPS:0,1,2,199",
            ],
        ],
        ConfigKeys.GEN_KW: [
            [
                duplicate_name,
                "FAULT_TEMPLATE",
                "MULTFLT.INC",
                "MULTFLT.TXT",
                "FORWARD_INIT:FALSE",
            ]
        ],
    }
    with pytest.raises(
        ConfigValidationError,
        match="GEN_KW and GEN_DATA contained duplicate name: Test_name",
    ):
        EnsembleConfig.from_dict(config_dict=config_dict)


def test_that_empty_grid_file_raises(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        FIELD foo bar out.roff
        GRID grid.GRDECL
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("grid.GRDECL", "w", encoding="utf-8") as fh:
            fh.writelines("")

        with pytest.raises(
            expected_exception=ConfigValidationError,
            match="did not contain dimensions",
        ):
            _ = ErtConfig.from_file("config.ert")
