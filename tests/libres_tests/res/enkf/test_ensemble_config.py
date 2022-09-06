import pytest

from ert._c_wrappers.enkf import ConfigKeys, EnsembleConfig
from ert._c_wrappers.enkf.enums import GenDataFileType


def test_create():
    conf = EnsembleConfig()
    assert len(conf) == 0
    assert "XYZ" not in conf

    with pytest.raises(KeyError):
        # pylint: disable=pointless-statement
        conf["KEY"]


def test_ensemble_config_constructor(setup_case):
    res_config = setup_case("local/configuration_tests", "ensemble_config.ert")
    assert res_config.ensemble_config == EnsembleConfig(
        config_dict={
            ConfigKeys.GEN_KW_TAG_FORMAT: "<%s>",
            ConfigKeys.GEN_DATA: [
                {
                    ConfigKeys.NAME: "SNAKE_OIL_OPR_DIFF",
                    ConfigKeys.INPUT_FORMAT: GenDataFileType.ASCII,
                    ConfigKeys.RESULT_FILE: "snake_oil_opr_diff_%d.txt",
                    ConfigKeys.REPORT_STEPS: [199],
                    ConfigKeys.INIT_FILES: None,
                    ConfigKeys.ECL_FILE: None,
                    ConfigKeys.TEMPLATE: None,
                    ConfigKeys.KEY_KEY: None,
                },
                {
                    ConfigKeys.NAME: "SNAKE_OIL_GPR_DIFF",
                    ConfigKeys.INPUT_FORMAT: GenDataFileType.ASCII,
                    ConfigKeys.RESULT_FILE: "snake_oil_gpr_diff_%d.txt",
                    ConfigKeys.REPORT_STEPS: [199],
                    ConfigKeys.INIT_FILES: None,
                    ConfigKeys.ECL_FILE: None,
                    ConfigKeys.TEMPLATE: None,
                    ConfigKeys.KEY_KEY: None,
                },
            ],
            ConfigKeys.GEN_KW: [
                {
                    ConfigKeys.NAME: "MULTFLT",
                    ConfigKeys.TEMPLATE: "FAULT_TEMPLATE",
                    ConfigKeys.OUT_FILE: "MULTFLT.INC",
                    ConfigKeys.PARAMETER_FILE: "MULTFLT.TXT",
                    ConfigKeys.INIT_FILES: None,
                    ConfigKeys.MIN_STD: None,
                    ConfigKeys.FORWARD_INIT: False,
                }
            ],
            ConfigKeys.SURFACE_KEY: [
                {
                    ConfigKeys.NAME: "TOP",
                    ConfigKeys.INIT_FILES: "surface/small.irap",
                    ConfigKeys.OUT_FILE: "surface/small_out.irap",
                    ConfigKeys.BASE_SURFACE_KEY: ("surface/small.irap"),
                    ConfigKeys.MIN_STD: None,
                    ConfigKeys.FORWARD_INIT: False,
                }
            ],
            ConfigKeys.SUMMARY: ["WOPR:OP_1"],
            ConfigKeys.FIELD_KEY: [
                {
                    ConfigKeys.NAME: "PERMX",
                    ConfigKeys.VAR_TYPE: "PARAMETER",
                    ConfigKeys.INIT_FILES: "fields/permx%d.grdecl",
                    ConfigKeys.OUT_FILE: "permx.grdcel",
                    ConfigKeys.ENKF_INFILE: None,
                    ConfigKeys.INIT_TRANSFORM: None,
                    ConfigKeys.OUTPUT_TRANSFORM: None,
                    ConfigKeys.INPUT_TRANSFORM: None,
                    ConfigKeys.MIN_KEY: None,
                    ConfigKeys.MAX_KEY: None,
                    ConfigKeys.MIN_STD: None,
                    ConfigKeys.FORWARD_INIT: False,
                }
            ],
            ConfigKeys.SCHEDULE_PREDICTION_FILE: [
                {
                    ConfigKeys.TEMPLATE: "input/schedule.sch",
                    ConfigKeys.INIT_FILES: None,
                    ConfigKeys.MIN_STD: None,
                    ConfigKeys.PARAMETER_KEY: None,
                }
            ],
        },
        grid=res_config.ecl_config.getGrid(),
    )
