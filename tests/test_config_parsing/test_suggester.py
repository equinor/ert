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
    assert suggestions[0].startswith("The UMASK keyword has been removed")
