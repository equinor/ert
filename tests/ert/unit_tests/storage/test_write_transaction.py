from pathlib import Path
from unittest.mock import mock_open, patch

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.storage._write_transaction import write_transaction


@pytest.mark.usefixtures("use_tmpdir")
@given(st.binary())
def test_write_transaction(data):
    filepath = Path("./file.txt")
    write_transaction(filepath, data)

    assert filepath.read_bytes() == data


def test_write_transaction_failure(tmp_path):
    with patch("builtins.open", mock_open()) as m:
        handle = m()

        def ctrlc(_):
            raise RuntimeError()

        handle.write = ctrlc

        path = tmp_path / "file.txt"
        with pytest.raises(RuntimeError):
            write_transaction(path, b"deadbeaf")

        assert not [
            c
            for c in m.mock_calls
            if path in c.args
            or str(path) in c.args
            or c.kwargs.get("file") in [path, str(path)]
        ], "There should be no calls opening the file when an write encounters a RuntimeError"


def test_write_transaction_overwrites(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("abc")
    write_transaction(path, b"deadbeaf")
    assert path.read_bytes() == b"deadbeaf"
