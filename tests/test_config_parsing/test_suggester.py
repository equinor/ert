import pytest

from ert._c_wrappers.config import ConfigParser
from ert._c_wrappers.enkf._deprecation_migration_suggester import (
    DeprecationMigrationSuggester,
)


@pytest.fixture
def suggester():
    return DeprecationMigrationSuggester(ConfigParser())


def test_that_suggester_gives_umask_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nUMASK 0222\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert suggestions[0].startswith("The UMASK keyword no longer")


def test_that_suggester_gives_havana_fault_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nHAVANA_FAULT\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert "The behavior of HAVANA_FAULT can be reproduced using" in suggestions[0]


def test_that_suggester_gives_multflt_migration(suggester, tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nMULTFLT\n")
    suggestions = suggester.suggest_migrations(str(tmp_path / "config.ert"))

    assert len(suggestions) == 1
    assert (
        "ert.readthedocs.io/en/latest/reference/configuration/keywords.html#gen-kw"
        in suggestions[0]
    )


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
