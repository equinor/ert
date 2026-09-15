import logging
import sys
from pathlib import Path

import pytest

from fmudesign.fmudesignrunner import main


@pytest.mark.slow
def test_that_log_fmudesign_logs_does_not_create_logs_folder(
    use_tmpdir, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(sys, "argv", ["fmudesign", "run", "foo.xlsx"])
    with pytest.raises(SystemExit):
        main()
    assert "Running fmudesign" in caplog.text
    assert "Input file foo.xlsx does not exist" in caplog.text
    assert not Path("logs").exists()
