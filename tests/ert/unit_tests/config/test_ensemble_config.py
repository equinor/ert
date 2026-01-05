from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ConfigValidationError, EnsembleConfig, ErtConfig
from ert.config.parsing import ConfigKeys


def test_create():
    empty_ens_conf = EnsembleConfig()
    conf_from_dict = EnsembleConfig.from_dict({})

    assert empty_ens_conf == conf_from_dict
    assert not conf_from_dict.parameters

    assert "XYZ" not in conf_from_dict

    with pytest.raises(KeyError):
        _ = conf_from_dict["KEY"]


@pytest.mark.usefixtures("use_tmpdir")
def test_ensemble_config_fails_on_non_sensical_grid_file():
    grid_file = Path("CEST_PAS_UNE_GRID")
    grid_file.write_text("a_grid_maybe?", encoding="utf-8")
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=r"Only EGRID and GRID formats are supported",
    ):
        EnsembleConfig.from_dict(config_dict={ConfigKeys.GRID: grid_file})


@pytest.mark.usefixtures("use_tmpdir")
def test_ensemble_config_duplicate_node_names():
    duplicate_name = "Test_name"
    Path("FAULT_TEMPLATE").write_text("", encoding="utf-8")
    config_dict = {
        ConfigKeys.GEN_DATA: [
            [
                duplicate_name,
                {
                    "RESULT_FILE": "snake_oil_opr_diff_%d.txt",
                    "REPORT_STEPS": "0,1,2,199",
                },
            ],
        ],
        ConfigKeys.GEN_KW: [
            [
                duplicate_name,
                ("FAULT_TEMPLATE", ""),
                "MULTFLT.INC",
                ("MULTFLT.TXT", f"{duplicate_name} UNIFORM 0 1"),
                {"FORWARD_INIT": "FALSE"},
            ]
        ],
    }
    with pytest.warns(match="The template file .* is empty"):
        EnsembleConfig.get_gen_kw_templates(config_dict)
    with pytest.raises(
        ValueError,
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
            match=r"Only EGRID and GRID formats are supported",
        ):
            _ = ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
def test_validation_of_duplicate_gen_kw_parameter_names():
    Path("FAULT_TEMPLATE").write_text("", encoding="utf-8")
    config_dict = {
        ConfigKeys.GEN_KW: [
            [
                "test_group1",
                ("FAULT_TEMPLATE", ""),
                "MULTFLT.INC",
                ("MULTFLT1.TXT", "a UNIFORM 0 1\nc UNIFORM 2 5"),
                {"FORWARD_INIT": "FALSE"},
            ],
            [
                "test_group2",
                ("FAULT_TEMPLATE", ""),
                "MULTFLT.INC",
                ("MULTFLT2.TXT", "a UNIFORM 0 1\nc UNIFORM 4 7"),
                {"FORWARD_INIT": "FALSE"},
            ],
        ],
    }
    with pytest.raises(
        ConfigValidationError,
        match=r"GEN_KW parameter names must be unique,"
        r" found duplicates: a\(2\), c\(2\)",
    ):
        ErtConfig.from_dict(config_dict=config_dict)
