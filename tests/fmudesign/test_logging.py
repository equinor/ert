import re
from pathlib import Path
from subprocess import CalledProcessError

import pytest

from .test_use_cases import _run_cli


@pytest.mark.slow
def test_that_log_folder_is_instantiated_with_fmudesign_cli_entrypoint(
    use_tmpdir, monkeypatch
):
    with pytest.raises(CalledProcessError):
        _run_cli("run", "does_not_exist.xlsx")
    assert Path("logs").exists()

    file = next(Path("logs").iterdir())

    # Implicitly test that underscores are replaced with dashes in the log file name
    log_file_pattern = (
        r"fmudesign-log-does-not-exist-xlsx-(\d{4})-(\d{2})-(\d{2})T\d{4}[+-]\d{4}\.txt"
    )
    assert re.match(log_file_pattern, file.name)
