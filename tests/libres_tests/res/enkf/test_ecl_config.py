#!/usr/bin/env python
import os.path
from datetime import datetime

from ecl.summary import EclSum

from ert._c_wrappers.enkf import ConfigKeys, EclConfig
from ert._c_wrappers.util import UIReturn


def test_grid(source_root):
    grid_file = source_root / "test-data/snake_oil_field/grid/CASE.EGRID"
    smspec_file = (
        source_root / "test-data/snake_oil_field/refcase/SNAKE_OIL_FIELD.SMSPEC"
    )
    ec = EclConfig()
    ui = ec.validateGridFile(str(grid_file))
    assert ui
    assert isinstance(ui, UIReturn)

    ui = ec.validateGridFile("Does/Not/Exist")
    assert not ui

    assert os.path.exists(smspec_file)
    ui = ec.validateGridFile(str(smspec_file))
    assert not ui


def test_datafile(source_root):
    ec = EclConfig()
    ui = ec.validateDataFile("DoesNotExist")
    assert not ui

    dfile = str(source_root / "test-data/eclipse/SPE1.DATA")
    ui = ec.validateDataFile(dfile)
    assert ui
    ec.setDataFile(dfile)
    assert dfile == ec.getDataFile()


def test_refcase(source_root):
    ec = EclConfig()
    dfile = str(source_root / "test-data/snake_oil/refcase/SNAKE_OIL_FIELD")

    ui = ec.validateRefcase("Does/not/exist")
    assert not ui

    ui = ec.validateRefcase(dfile)
    assert ui
    ec.loadRefcase(dfile)
    refcase = ec.getRefcase()
    assert isinstance(refcase, EclSum)
    refcaseName = ec.getRefcaseName()
    assert dfile == refcaseName


def test_wrongly_configured_refcase_path():
    config_dict = {
        ConfigKeys.REFCASE: "this/is/not/REFCASE",
    }
    ecl_config = EclConfig(config_dict=config_dict)
    assert ecl_config.getRefcaseName() is None
    assert ecl_config.getRefcase() is None


def test_ecl_config_constructor(setup_case):
    res_config = setup_case("configuration_tests", "ecl_config.ert")
    assert res_config.ecl_config == EclConfig(
        config_dict={
            ConfigKeys.DATA_FILE: "input/SPE1.DATA",
            ConfigKeys.ECLBASE: "input/<ECLIPSE_NAME>-%d",
            ConfigKeys.GRID: "input/CASE.EGRID",
            ConfigKeys.REFCASE: "input/refcase/SNAKE_OIL_FIELD",
            ConfigKeys.SCHEDULE_PREDICTION_FILE: "input/schedule.sch",
        }
    )


def test_that_refcase_gets_correct_name(tmpdir):
    config_dict = {
        ConfigKeys.REFCASE: "MY_REFCASE",
    }

    with tmpdir.as_cwd():
        # Create a refcase
        ecl_sum = EclSum.writer("MY_REFCASE", datetime(2014, 9, 10), 10, 10, 10)
        ecl_sum.addVariable("FOPR", unit="SM3/DAY")
        t_step = ecl_sum.addTStep(2, sim_days=1)
        t_step["FOPR"] = 1
        ecl_sum.fwrite()

        ecl_config = EclConfig(config_dict=config_dict)
        assert ecl_config.getRefcaseName() == tmpdir / "MY_REFCASE"
