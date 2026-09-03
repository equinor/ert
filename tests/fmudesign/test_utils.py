import string
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given

from fmudesign.utils import resolve_path


@pytest.mark.usefixtures("use_tmpdir")
@given(st.text(alphabet=string.ascii_letters))
def test_that_resolve_path_resolves_any_file_extension(file_extension):
    filename = f"foo.{file_extension}"
    Path(filename).touch()
    resolved = resolve_path(filename)
    assert resolved == str(Path(filename).resolve())


def test_that_resolve_path_resolves_relative_path_to_base_file(use_tmpdir):
    folder = "path/going/down/"
    Path(folder).mkdir(parents=True)

    base_file = folder + "design.xlsx"
    Path(base_file).touch()

    relative_file = "seeds.txt"
    Path(folder + relative_file).touch()

    resolved = resolve_path(relative_file, base_file=base_file)
    assert resolved == str((Path(folder) / relative_file).resolve())
