import warnings
from unittest.mock import patch

import pytest

from ert.config import ConfigWarning, ErtConfig
from ert.config.forward_model_step import (
    UserInstalledForwardModelStep,
)
from ert.config.parsing.config_schema_deprecations import (
    JUST_REMOVE_KEYWORDS,
    REPLACE_WITH_GEN_KW,
    RSH_KEYWORDS,
    USE_QUEUE_OPTION,
)
from ert.config.parsing.deprecation_info import DeprecationInfo


def test_is_angle_bracketed_detects_words_surrounded_by_brackets():
    assert DeprecationInfo.is_angle_bracketed("<KEY>")
    assert not DeprecationInfo.is_angle_bracketed("KEY")
    assert not DeprecationInfo.is_angle_bracketed("K<E>Y")
    assert not DeprecationInfo.is_angle_bracketed("")


@pytest.mark.parametrize("kw", JUST_REMOVE_KEYWORDS)
def test_that_a_warning_is_shown_for_keys_that_can_just_be_removed(kw):
    with pytest.warns(ConfigWarning, match=f"The keyword {kw} no longer"):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


def test_that_a_deprecation_warning_is_shown_if_the_havana_fault_keyword_is_used():
    with pytest.warns(
        ConfigWarning, match="The behavior of HAVANA_FAULT can be reproduced using"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nHAVANA_FAULT\n")


@pytest.mark.parametrize("kw", REPLACE_WITH_GEN_KW)
def test_that_a_link_to_the_docs_is_shown_for_keywords_that_can_be_replaced_by_gen_kw(
    kw,
):
    with pytest.warns(
        ConfigWarning,
        match="ert.readthedocs.io/en/latest/reference/configuration/keywords.html#gen-kw",
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


@pytest.mark.parametrize("kw", RSH_KEYWORDS)
def test_that_a_deprecation_is_shown_for_keywords_related_to_rhs_queues(kw):
    with pytest.warns(
        ConfigWarning, match="deprecated and removed support for RSH queues"
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


@pytest.mark.parametrize("kw", USE_QUEUE_OPTION)
def test_that_migration_info_is_shown_for_keywords_replaced_by_the_queue_option_keyword(
    kw,
):
    with pytest.warns(
        ConfigWarning, match=f"The {kw} keyword has been removed. For most cases "
    ):
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{kw}\n")


def test_that_a_deprecation_message_is_shown_when_the_refcase_list_keyword_is_used():
    with pytest.warns(
        ConfigWarning,
        match="The corresponding plotting functionality was removed in 2015",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nREFCASE_LIST case.DATA\n")


def test_that_a_deprecation_message_is_shown_when_the_rftpath_keyword_is_used():
    with pytest.warns(
        ConfigWarning,
        match="The corresponding plotting functionality was removed in 2015",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRFTPATH rfts/\n")


def test_that_a_deprecation_message_is_shown_when_the_end_date_keyword_is_used():
    with pytest.warns(
        ConfigWarning, match="only display a warning in case of problems"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nEND_DATE 2023.01.01\n")


def test_that_a_deprecation_message_is_shown_when_the_rerun_start_keyword_is_used():
    with pytest.warns(
        ConfigWarning, match="used for the deprecated run mode ENKF_ASSIMILATION"
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRERUN_START 2023.01.01\n")


def test_that_a_deprecation_message_is_shown_when_the_delete_runpath_keyword_is_used():
    with pytest.warns(ConfigWarning, match="It was removed in 2017"):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nDELETE_RUNPATH TRUE\n")


def test_that_using_printf_format_value_placeholders_in_runpath_is_deprecated():
    with pytest.warns(
        ConfigWarning,
        match="RUNPATH keyword contains deprecated value"
        r" placeholders: %d, instead use: .*real-<IENS>\/iter-<ITER>",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n")


@pytest.mark.filterwarnings("error")
def test_that_no_deprecation_warning_is_shown_for_substitution_value_placeholders():
    ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\nRUNPATH real-<IENS>/iter-<ITER>\n"
    )


def test_that_a_deprecation_message_is_shown_when_the_plot_settings_keyword_is_used():
    with pytest.warns(
        ConfigWarning,
        match="The keyword PLOT_SETTINGS was removed in 2019 and has no effect",
    ):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nPLOT_SETTINGS some args\n")


def test_that_a_deprecation_message_is_shown_when_the_update_settings_keyword_is_used():
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
            "Using .* with substitution strings that are not of the form "
            "'<KEY>' is deprecated. "
            "Please change A to <A>",
        ),
        (
            "<A<B>> C",
            "Using .* with substitution strings that are not of the form "
            "'<KEY>' is deprecated. "
            "Please change <A<B>> to <AB>",
        ),
        (
            "<A><B> C",
            "Using .* with substitution strings that are not of the "
            "form '<KEY>' is deprecated. "
            "Please change <A><B> to <AB>",
        ),
    ],
)
def test_that_a_deprecation_message_is_shown_for_substitutions_without_brackets(
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


def test_when_creating_list_of_deprecation_warnings_pre_defines_are_taken_into_account(
    tmp_path,
):
    """This is a regression test for a bug where the code that created
    deprecation warnings did not take into account pre-defines like
    <CONFIG_PATH>.
    """
    (tmp_path / "workflow").write_text("")
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nLOAD_WORKFLOW <CONFIG_PATH>/workflow\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ErtConfig.from_file(tmp_path / "config.ert")


def test_that_a_deprecation_message_is_shown_for_the_schedule_prediction_file_keyword():
    with pytest.warns(
        ConfigWarning,
        match="The 'SCHEDULE_PREDICTION_FILE' config keyword has been removed",
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nSCHEDULE_PREDICTION_FILE no no no\n"
        )


def test_that_a_deprecation_message_is_shown_for_use_of_the_job_prefix_queue_option():
    with pytest.warns(
        ConfigWarning,
        match="JOB_PREFIX as QUEUE_OPTION to the TORQUE system is deprecated",
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nQUEUE_OPTION TORQUE JOB_PREFIX foo\n"
        )


def test_that_forward_model_design2params_is_deprecated():
    # Create a mock DESIGN2PARAMS forward model step, since it is not installed
    mock_design2params_step = UserInstalledForwardModelStep(
        name="DESIGN2PARAMS",
        executable="design2params",
    )

    with (
        patch(
            "ert.config.ert_config.installed_forward_model_steps_from_dict",
            return_value={"DESIGN2PARAMS": mock_design2params_step},
        ),
        pytest.warns(
            ConfigWarning,
            match="FORWARD_MODEL DESIGN2PARAMS will be replaced with DESIGN_MATRIX. "
            "Note that validation of DESIGN_MATRIX is more strict, missing values.*",
        ),
    ):
        config = (
            "FORWARD_MODEL DESIGN2PARAMS(<xls_filename>=poly_design.xslx, "
            "<designsheet>=TheDesignSheet, <defaultssheet>=TheDefaultsSheet)"
        )
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 1\n{config}")
