import warnings

import pytest

from ert.config import ConfigWarning, ErtConfig
from ert.config.parsing.config_schema_deprecations import (
    JUST_REMOVE_KEYWORDS,
    REPLACE_WITH_GEN_KW,
    RSH_KEYWORDS,
    USE_QUEUE_OPTION,
)


def make_suggestion_list(path):
    with warnings.catch_warnings(record=True) as all_warnings:
        _ = ErtConfig.from_file(path)
    return [
        str(w.message)
        for w in all_warnings
        if w.category == ConfigWarning and w.message.info.is_deprecation
    ]


@pytest.mark.parametrize("kw", JUST_REMOVE_KEYWORDS)
def test_that_suggester_gives_simple_migrations(tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(f"The keyword {kw} no longer" in s for s in suggestions)


def test_that_suggester_gives_havana_fault_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nHAVANA_FAULT\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The behavior of HAVANA_FAULT can be reproduced using" in s for s in suggestions
    )


@pytest.mark.parametrize("kw", REPLACE_WITH_GEN_KW)
def test_that_suggester_gives_gen_kw_migrations(tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "ert.readthedocs.io/en/latest/reference/configuration/keywords.html#gen-kw" in s
        for s in suggestions
    )


@pytest.mark.parametrize("kw", RSH_KEYWORDS)
def test_that_suggester_gives_rsh_migrations(tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "deprecated and removed support for RSH queues." in s for s in suggestions
    )


@pytest.mark.parametrize("kw", USE_QUEUE_OPTION)
def test_that_suggester_gives_queue_option_migrations(tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        f"The {kw} keyword has been removed. For most cases " in s for s in suggestions
    )


def test_that_suggester_gives_refcase_list_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nREFCASE_LIST case.DATA\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The corresponding plotting functionality was removed in 2015" in s
        for s in suggestions
    )


def test_that_suggester_gives_rftpath_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nRFTPATH rfts/\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The corresponding plotting functionality was removed in 2015" in s
        for s in suggestions
    )


def test_that_suggester_gives_end_date_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nEND_DATE 2023.01.01\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any("only display a warning in case of problems" in s for s in suggestions)


def test_that_suggester_gives_rerun_start_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nRERUN_START 2023.01.01\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "used for the deprecated run mode ENKF_ASSIMILATION" in s for s in suggestions
    )


def test_that_suggester_gives_delete_runpath_migration(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nDELETE_RUNPATH TRUE\n")
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any("It was removed in 2017" in s for s in suggestions)


def test_suggester_gives_runpath_deprecated_specifier_migration(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n"
    )
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "RUNPATH keyword contains deprecated value placeholders" in s
        for s in suggestions
    )


def test_suggester_gives_no_runpath_deprecated_specifier_migration(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nRUNPATH real-<IENS>/iter-<ITER>\n"
    )
    no_suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    (tmp_path / "config.wrong.ert").write_text(
        "NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n"
    )
    suggestions = make_suggestion_list(str(tmp_path / "config.wrong.ert"))

    assert not any(
        "RUNPATH keyword contains deprecated value placeholders" in s
        for s in no_suggestions
    ) and any(
        "RUNPATH keyword contains deprecated value placeholders" in s
        for s in suggestions
    )


def test_suggester_gives_plot_settings_migration(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nPLOT_SETTINGS some args\n"
    )
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The keyword PLOT_SETTINGS was removed in 2019 and has no effect" in s
        for s in suggestions
    )


def test_suggester_gives_update_settings_migration(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nUPDATE_SETTINGS some args\n"
    )
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The UPDATE_SETTINGS keyword has been removed and no longer" in s
        for s in suggestions
    )


def test_suggester_gives_deprecated_define_migration_hint(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\n"
        "DEFINE <KEY1> x1\n"
        "DEFINE A B\n"
        "DEFINE <A<B>> C\n"
        "DEFINE <A><B> C\n"
    )
    for suggestion, expected in zip(
        make_suggestion_list(str(tmp_path / "config.ert")),
        [
            " Please change A to <A>",
            " Please change <A<B>> to <AB>",
            " Please change <A><B> to <AB>",
        ],
    ):
        assert (
            "Using DEFINE with substitution strings"
            " that are not of the form '<KEY>' is deprecated." in suggestion
        )
        assert suggestion.endswith(expected)


def test_suggester_does_not_report_non_existent_path_due_to_missing_pre_defines(
    tmp_path,
):
    (tmp_path / "workflow").write_text("")
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nLOAD_WORKFLOW <CONFIG_PATH>/workflow\n"
    )
    assert [
        x
        for x in make_suggestion_list(str(tmp_path / "config.ert"))
        if "DATA_KW" not in x and "DEFINE" not in x
    ] == []


def test_that_suggester_gives_schedule_prediciton_migration(tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nSCHEDULE_PREDICTION_FILE no no no\n"
    )
    suggestions = make_suggestion_list(str(tmp_path / "config.ert"))

    assert any(
        "The 'SCHEDULE_PREDICTION_FILE' config keyword has been removed" in s
        for s in suggestions
    )
