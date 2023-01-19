import pytest

from ert._c_wrappers.enkf._deprecation_migration_suggester import (
    DeprecationMigrationSuggester,
)
from ert._c_wrappers.enkf.res_config import ResConfig


@pytest.fixture
def suggester():
    return DeprecationMigrationSuggester(ResConfig._create_user_config_parser())


@pytest.mark.parametrize("kw", DeprecationMigrationSuggester.JUST_REMOVE_KEYWORDS)
def test_that_suggester_gives_simple_migrations(suggester, tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert suggestions[0].startswith(f"The keyword {kw} no longer")


def test_that_suggester_gives_havana_fault_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nHAVANA_FAULT\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "The behavior of HAVANA_FAULT can be reproduced using" in suggestions[0]


@pytest.mark.parametrize("kw", DeprecationMigrationSuggester.REPLACE_WITH_GEN_KW)
def test_that_suggester_gives_gen_kw_migrations(suggester, tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert (
        "ert.readthedocs.io/en/latest/reference/configuration/keywords.html#gen-kw"
        in suggestions[0]
    )


@pytest.mark.parametrize("kw", DeprecationMigrationSuggester.RSH_KEYWORDS)
def test_that_suggester_gives_rsh_migrations(suggester, tmp_path, kw):
    (tmp_path / "config.ert").write_text(f"NUM_REALIZATIONS 1\n{kw}\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "deprecated and removed support for RHS queues." in suggestions[0]


def test_that_suggester_gives_refcase_list_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nREFCASE_LIST case.DATA\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert (
        "The corresponding plotting functionality was removed in 2015" in suggestions[0]
    )


def test_that_suggester_gives_rftpath_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nRFTPATH rfts/\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert (
        "The corresponding plotting functionality was removed in 2015" in suggestions[0]
    )


def test_that_suggester_gives_end_date_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nEND_DATE 2023.01.01\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "only display a warning in case of problems" in suggestions[0]


def test_that_suggester_gives_rerun_start_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nRERUN_START 2023.01.01\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "used for the deprecated run mode ENKF_ASSIMILATION" in suggestions[0]


def test_that_suggester_gives_delete_runpath_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nDELETE_RUNPATH TRUE\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "It was removed in 2017" in suggestions[0]


def test_suggester_gives_runpath_deprecated_specifier_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nRUNPATH real-%d/iter-%d\n"
    )
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "RUNPATH keyword contains deprecated value placeholders" in suggestions[0]


def test_suggester_gives_no_runpath_deprecated_specifier_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nRUNPATH real-<IENS>/iter-<ITER>\n"
    )
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 0


def test_suggester_gives_plot_settings_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text(
        "NUM_REALIZATIONS 1\nPLOT_SETTINGS some args\n"
    )
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert (
        "The PLOT_SETTINGS keyword was once used to set up parameters" in suggestions[0]
    )
