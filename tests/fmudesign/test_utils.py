import string
from pathlib import Path

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given

from fmudesign.utils import map_dependencies, resolve_path


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


@pytest.mark.parametrize(
    "from_values",
    [
        pytest.param(["C1", "C1"], id="repeated-text"),
        pytest.param([1, "1.0"], id="equivalent-numeric-values"),
    ],
)
def test_that_duplicate_normalized_dependency_keys_raise_value_error(from_values):
    dependencies = {
        "SOURCE": {
            "from_values": from_values,
            "to_params": {"TARGET": ["first", "second"]},
        }
    }

    with pytest.raises(ValueError, match="Duplicate dependency keys for 'SOURCE'"):
        map_dependencies(
            pd.DataFrame({"SOURCE": from_values[:1]}), dependencies=dependencies
        )


def test_that_numeric_dependency_lookups_and_copies_preserve_realization_order():
    result = map_dependencies(
        pd.DataFrame({"SOURCE": [2, 1, 2]}),
        dependencies={
            "SOURCE": {
                "from_values": ["1", 2],
                "to_params": {"COPY": [], "TARGET": [10, "20"]},
            }
        },
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": [2, 1, 2],
        "COPY": [2, 1, 2],
        "TARGET": [20, 10, 20],
    }
