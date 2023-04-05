import os
from datetime import datetime

import pytest
from ecl.grid.ecl_grid import EclGrid
from ecl.summary import EclSum

from ert.parsing import ConfigWarning, ConfigValidationError
from ert._c_wrappers.enkf import ConfigKeys, EnsembleConfig, ErtConfig
from ert._c_wrappers.enkf.enums import ErtImplType, GenDataFileType


def test_create():
    empty_ens_conf = EnsembleConfig()
    conf_from_dict = EnsembleConfig.from_dict({})

    assert empty_ens_conf == conf_from_dict
    assert conf_from_dict.get_refcase_file is None
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
        EnsembleConfig.from_dict(config_dict=config_dict)


def test_ensemble_config_construct_refcase_and_grid(setup_case):
    setup_case("configuration_tests", "ensemble_config.ert")
    grid_file = "grid/CASE.EGRID"
    refcase_file = "input/refcase/SNAKE_OIL_FIELD"

    ec = EnsembleConfig.from_dict(
        config_dict={
            ConfigKeys.GRID: grid_file,
            ConfigKeys.REFCASE: refcase_file,
        },
    )

    assert isinstance(ec, EnsembleConfig)
    assert isinstance(ec.grid, EclGrid)
    assert isinstance(ec.refcase, EclSum)

    assert ec._grid_file == os.path.realpath(grid_file)
    assert ec._refcase_file == os.path.realpath(refcase_file)


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
def test_that_files_for_refcase_exists(setup_case, existing_suffix, expected_suffix):
    setup_case("configuration_tests", "ensemble_config.ert")
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


@pytest.mark.parametrize(
    "gen_data_str, expected",
    [
        pytest.param(
            "GDK RESULT_FILE:Results INPUT_FORMAT:ASCII REPORT_STEPS:10",
            None,
            id="RESULT_FILE missing %d in file name",
        ),
        pytest.param(
            "GDK RESULT_FILE:Results%d INPUT_FORMAT:ASCII",
            None,
            id="REPORT_STEPS missing",
        ),
        pytest.param(
            "GDK RESULT_FILE:Results%d INPUT_FORMAT:ASCII REPORT_STEPS:10,20,30",
            "Valid",
            id="Valid case",
        ),
    ],
)
def test_gen_data_node(gen_data_str, expected):
    node = EnsembleConfig.gen_data_node(gen_data_str.split(" "))
    if expected is None:
        assert node == expected
    else:
        assert node is not None
        assert node.getImplementationType() == ErtImplType.GEN_DATA
        assert node.getDataModelConfig().getNumReportStep() == 3
        assert node.getDataModelConfig().hasReportStep(10)
        assert node.getDataModelConfig().hasReportStep(20)
        assert node.getDataModelConfig().hasReportStep(30)
        assert not node.getDataModelConfig().hasReportStep(32)
        assert node.getDataModelConfig().getInputFormat() == GenDataFileType.ASCII


@pytest.mark.parametrize(
    "input_format",
    [
        pytest.param(
            "INPUT_FORMAT:ASCIIX",
            id="Unsupported INPUT_FORMAT",
        ),
        pytest.param(
            "INPUT_FORMAT:",
            id="Missing INPUT_FORMAT no format",
        ),
        pytest.param(
            "",
            id="Missing INPUT_FORMAT no key and format",
        ),
    ],
)
def test_gen_data_node_input_format(input_format):
    gen_data_str = f"GDK RESULT_FILE:Results%d {input_format} REPORT_STEPS:10,20,30"

    with pytest.warns(
        ConfigWarning,
        match="Missing or unsupported GEN_DATA INPUT_FORMAT for key 'GDK'.",
    ):
        node = EnsembleConfig.gen_data_node(gen_data_str.split(" "))
        assert node is not None
        assert node.getImplementationType() == ErtImplType.GEN_DATA
        assert node.getDataModelConfig().getNumReportStep() == 3


def test_get_surface_node(setup_case):
    _ = setup_case("configuration_tests", "ensemble_config.ert")
    surface_str = "TOP"
    with pytest.raises(
        ConfigValidationError, match="Missing INIT_FILES:/path/to/input/files"
    ):
        EnsembleConfig.get_surface_node(surface_str.split(" "))

    surface_in = "surface/small.irap"
    surface_out = "surface/small_out.irap"
    # add init file
    surface_str += f" INIT_FILES:{surface_in}"

    with pytest.raises(
        ConfigValidationError, match="Missing OUTPUT_FILE:/path/to/output_file"
    ):
        EnsembleConfig.get_surface_node(surface_str.split(" "))

    # add output file
    surface_str += f" OUTPUT_FILE:{surface_out}"
    with pytest.raises(
        ConfigValidationError, match="Missing BASE_SURFACE:/path/to/base_surface_file"
    ):
        EnsembleConfig.get_surface_node(surface_str.split(" "))

    # add base surface
    surface_str += f" BASE_SURFACE:{surface_in}"

    surface_node = EnsembleConfig.get_surface_node(surface_str.split(" "))

    assert surface_node is not None

    surface_str += " FORWARD_INIT:TRUE"
    surface_node = EnsembleConfig.get_surface_node(surface_str.split(" "))


def test_surface_bad_init_values(setup_case):
    _ = setup_case("configuration_tests", "ensemble_config.ert")
    surface_in = "path/42"
    surface_out = "surface/small_out.irap"
    surface_str = (
        f"TOP INIT_FILES:{surface_in}"
        f" OUTPUT_FILE:{surface_out}"
        f" BASE_SURFACE:{surface_in}"
    )
    error = (
        f"INIT_FILES:{surface_in!r} File not found"
        f" BASE_SURFACE:{surface_in!r} File not found "
    )
    with pytest.raises(ConfigValidationError, match=error):
        EnsembleConfig.get_surface_node(surface_str.split(" "))


def test_ensemble_config_duplicate_node_names(setup_case):
    _ = setup_case("configuration_tests", "ensemble_config.ert")
    duplicate_name = "Test_name"
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
    error_match = f"key {duplicate_name!r} already present in ensemble config"

    with pytest.raises(ConfigValidationError, match=error_match):
        EnsembleConfig.from_dict(config_dict=config_dict)


@pytest.mark.parametrize(
    "result_file, fail",
    [
        pytest.param(
            "RESULT_FILE:",
            True,
            id="RESULT_FILE key but no file",
        ),
        pytest.param(
            "",
            True,
            id="No RESULT_FILE key",
        ),
        pytest.param(
            'RESULT_FILE:"file_in_quotes_%d.out"',
            True,
            id="File in quotes",
        ),
        pytest.param(
            "RESULT_FILE:poly_%d.out",
            False,
            id="This should not fail",
        ),
    ],
)
def test_malformed_or_missing_gen_data_result_file(setup_case, result_file, fail):
    _ = setup_case("poly_example", "poly.ert")
    # Add extra GEN_DATA key to config file
    config_line = f"""
    GEN_DATA POLY_RES_2 {result_file} REPORT_STEPS:0 INPUT_FORMAT:ASCII
    """
    with open("poly.ert", "a", encoding="utf-8") as f:
        f.write(config_line)

    if fail:
        with pytest.raises(
            ConfigValidationError,
            match="Missing or unsupported RESULT_FILE for GEN_DATA",
        ):
            ErtConfig.from_file("poly.ert")
    else:
        ErtConfig.from_file("poly.ert")
