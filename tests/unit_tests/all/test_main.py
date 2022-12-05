from unittest.mock import Mock

import pytest

import ert.shared
from ert.__main__ import ert_parser
from ert.cli import TEST_RUN_MODE


@pytest.mark.parametrize("input_path", ["a/path/config.ert", "another/path/config.ert"])
def test_parsed_config(monkeypatch, input_path):
    monkeypatch.setattr(
        ert.__main__, "valid_file", Mock(side_effect=lambda _: input_path)
    )
    parsed = ert_parser(None, [TEST_RUN_MODE, input_path])
    assert parsed.config == input_path


def test_version_mocked(capsys, monkeypatch):
    monkeypatch.setattr(ert.shared, "__version__", "1.0.3")

    try:
        ert_parser(None, ["--version"])
    except SystemExit as e:
        assert e.code == 0

    ert_version, _ = capsys.readouterr()
    ert_version = ert_version.rstrip("\n")

    assert ert_version == "1.0.3"
