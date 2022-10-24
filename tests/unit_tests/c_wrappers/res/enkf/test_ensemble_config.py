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
    res_config = setup_case("configuration_tests", "ensemble_config.ert")
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
                    ConfigKeys.FORWARD_INIT: False,
                }
            ],
            ConfigKeys.SURFACE_KEY: [
                {
                    ConfigKeys.NAME: "TOP",
                    ConfigKeys.INIT_FILES: "surface/small.irap",
                    ConfigKeys.OUT_FILE: "surface/small_out.irap",
                    ConfigKeys.BASE_SURFACE_KEY: ("surface/small.irap"),
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
                    ConfigKeys.FORWARD_INIT: False,
                }
            ],
            ConfigKeys.SCHEDULE_PREDICTION_FILE: [
                {
                    ConfigKeys.TEMPLATE: "input/schedule.sch",
                    ConfigKeys.INIT_FILES: None,
                    ConfigKeys.PARAMETER_KEY: None,
                }
            ],
            ConfigKeys.GRID: "grid/CASE.EGRID",  # ecl
            # ConfigKeys.REFCASE: "../input/refcase/SNAKE_OIL_FIELD",  # ecl
        },
    )


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
    with open(refcase_file, "w+", encoding="utf-8") as refcase_file_handler:
        refcase_file_handler.write(refcase_file_content)
    with pytest.raises(expected_exception=IOError, match=refcase_file):
        config_dict = {ConfigKeys.REFCASE: refcase_file}
        EnsembleConfig(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
def test_ensemble_config_fails_on_non_sensical_grid_file():
    grid_file = "BRICKWALL"
    # NB: this is just silly ASCII content, not even close to a correct GRID file
    grid_file_content = """
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|__
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|__
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
"""
    with open(grid_file, "w+", encoding="utf-8") as grid_file_handler:
        grid_file_handler.write(grid_file_content)
    with pytest.raises(expected_exception=ValueError, match=grid_file):
        config_dict = {ConfigKeys.GRID: grid_file}
        EnsembleConfig(config_dict=config_dict)
