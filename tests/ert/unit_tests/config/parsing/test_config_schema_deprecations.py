import warnings
from pathlib import Path

import pytest

from ert.config import ConfigWarning, ErtConfig
from ert.config.parsing.config_schema_deprecations import (
    JUST_REMOVE_KEYWORDS,
    REPLACE_WITH_GEN_KW,
    RSH_KEYWORDS,
    USE_QUEUE_OPTION,
)
from ert.config.parsing.deprecation_info import DeprecationInfo


def test_is_angle_bracketed():
    assert DeprecationInfo.is_angle_bracketed("<KEY>")
    assert not DeprecationInfo.is_angle_bracketed("KEY")
    assert not DeprecationInfo.is_angle_bracketed("K<E>Y")
    assert not DeprecationInfo.is_angle_bracketed("")


@pytest.mark.parametrize("kw", JUST_REMOVE_KEYWORDS)
def test_that_suggester_gives_simple_migrations(kw):
    with pytest.warns(ConfigWarning, match=f"The keyword {kw} no longer"):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


def test_that_suggester_gives_havana_fault_migration():
    with pytest.warns(
        ConfigWarning, match="The behavior of HAVANA_FAULT can be reproduced using"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nHAVANA_FAULT\n")


@pytest.mark.parametrize("kw", REPLACE_WITH_GEN_KW)
def test_that_suggester_gives_gen_kw_migrations(kw):
    with pytest.warns(
        ConfigWarning,
        match="ert.readthedocs.io/en/latest/reference/configuration/keywords.html#gen-kw",
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


@pytest.mark.parametrize("kw", RSH_KEYWORDS)
def test_that_suggester_gives_rsh_migrations(kw):
    with pytest.warns(
        ConfigWarning, match="deprecated and removed support for RSH queues"
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


@pytest.mark.parametrize("kw", USE_QUEUE_OPTION)
def test_that_suggester_gives_queue_option_migrations(kw):
    with pytest.warns(
        ConfigWarning, match=f"The {kw} keyword has been removed. For most cases "
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


def test_that_suggester_gives_refcase_list_migration():
    with pytest.warns(
        ConfigWarning,
        match="The corresponding plotting functionality was removed in 2015",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nREFCASE_LIST case.DATA\n")


def test_that_suggester_gives_rftpath_migration():
    with pytest.warns(
        ConfigWarning,
        match="The corresponding plotting functionality was removed in 2015",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRFTPATH rfts/\n")


def test_that_suggester_gives_end_date_migration():
    with pytest.warns(
        ConfigWarning, match="only display a warning in case of problems"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nEND_DATE 2023.01.01\n")


def test_that_suggester_gives_rerun_start_migration():
    with pytest.warns(
        ConfigWarning, match="used for the deprecated run mode ENKF_ASSIMILATION"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRERUN_START 2023.01.01\n")


def test_that_suggester_gives_delete_runpath_migration():
    with pytest.warns(ConfigWarning, match="It was removed in 2017"):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nDELETE_RUNPATH TRUE\n")


def test_suggester_gives_runpath_deprecated_specifier_migration():
    with pytest.warns(
        ConfigWarning,
        match="RUNPATH keyword contains deprecated value"
        r" placeholders: %d, instead use: .*real-<IENS>\/iter-<ITER>",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n")


def test_suggester_gives_no_runpath_deprecated_specifier_migration():
    # Assert no warnings from this line:  FIXMEEEE
    ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\nRUNPATH real-<IENS>/iter-<ITER>\n"
    )

    with pytest.warns(
        ConfigWarning, match="RUNPATH keyword contains deprecated value placeholders"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n")


def test_suggester_gives_plot_settings_migration():
    with pytest.warns(
        ConfigWarning,
        match="The keyword PLOT_SETTINGS was removed in 2019 and has no effect",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nPLOT_SETTINGS some args\n")


def test_suggester_gives_update_settings_migration():
    with pytest.warns(
        ConfigWarning,
        match="The UPDATE_SETTINGS keyword has been removed and no longer",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nUPDATE_SETTINGS some args\n")


@pytest.mark.parametrize("definer", ["DEFINE", "DATA_KW"])
@pytest.mark.parametrize(
    "definition, expected",
    [
        ("<KEY1> x1", None),
        (
            "A B",
            "Using .* with substitution strings that are not of the form '<KEY>' is deprecated. "
            "Please change A to <A>",
        ),
        (
            "<A<B>> C",
            "Using .* with substitution strings that are not of the form '<KEY>' is deprecated. "
            "Please change <A<B>> to <AB>",
        ),
        (
            "<A><B> C",
            "Using .* with substitution strings that are not of the form '<KEY>' is deprecated. "
            "Please change <A><B> to <AB>",
        ),
    ],
)
def test_suggester_gives_deprecated_define_migration_hint(
    definer, definition, expected
):
    contents = f"NUM_REALIZATIONS 1\n{definer} {definition}\n"
    print(f"{definer} {definition}")
    if expected:
        with pytest.warns(ConfigWarning, match=expected):
            ErtConfig.from_file_contents(contents)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ErtConfig.from_file_contents(contents)


def test_suggester_does_not_report_non_existent_path_due_to_missing_pre_defines(
    tmp_path,
):
    (tmp_path / "workflow").write_text("")
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nLOAD_WORKFLOW <CONFIG_PATH>/workflow\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ErtConfig.from_file(tmp_path / "config.ert")


def test_that_suggester_gives_schedule_prediciton_migration():
    with pytest.warns(
        ConfigWarning,
        match="The 'SCHEDULE_PREDICTION_FILE' config keyword has been removed",
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nSCHEDULE_PREDICTION_FILE no no no\n"
        )


def test_that_suggester_gives_job_prefix_migration():
    with pytest.warns(
        ConfigWarning,
        match="JOB_PREFIX as QUEUE_OPTION to the TORQUE system is deprecated",
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nQUEUE_OPTION TORQUE JOB_PREFIX foo\n"
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_gen_kw_with_init_files_warns():
    Path("coeff_priors").write_text("a UNIFORM 0 1", encoding="utf-8")
    Path("init_file").write_text("a 0.4", encoding="utf-8")
    with pytest.warns(
        ConfigWarning,
        match="GEN_KW with INIT_FILES is deprecated, and will be removed in the next version.",
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nGEN_KW COEFFS coeff_priors INIT_FILES:init_file%d"
        )
