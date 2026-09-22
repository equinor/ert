import string
from pathlib import Path

import hypothesis.strategies as st
import polars as pl
import pytest
import xlsxwriter
from hypothesis import given

from fmudesign.utils import (
    map_dependencies,
    parameters_from_extern,
    resolve_path,
    seeds_from_extern,
)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        pytest.param(
            '\n \t\nVALUE\n1\n\n""\n2\n\n',
            {"VALUE": [1, None, 2]},
            id="quoted-empty-field",
        ),
        pytest.param(
            "A,B\n\n1,first\n,\n \t\n2,last\n",
            {"A": [1, None, 2], "B": ["first", None, "last"]},
            id="empty-fields",
        ),
        pytest.param(
            'VALUE\n\n"first\n\nlast"\n \t\n" \t "\n"quote ""inside"", comma"\n',
            {"VALUE": ["first\n\nlast", " \t ", 'quote "inside", comma']},
            id="quoted-newlines-and-whitespace",
        ),
    ],
)
def test_that_external_csv_skips_blank_lines_but_preserves_empty_and_quoted_fields(
    tmp_path, contents, expected
):
    path = tmp_path / "parameters.csv"
    path.write_text(contents)

    assert parameters_from_extern(str(path)).to_dict(as_series=False) == expected


@pytest.mark.parametrize("suffix", ["xlsx", "csv", "txt"])
def test_that_seeds_from_extern_reads_first_column_as_integers(tmp_path, suffix):
    seeds_file = tmp_path / f"seeds.{suffix}"
    if suffix == "xlsx":
        with xlsxwriter.Workbook(seeds_file) as workbook:
            worksheet = workbook.add_worksheet()
            worksheet.write_string(0, 0, "2000")
            worksheet.write_number(1, 0, 2001)
            worksheet.write_number(3, 0, 2002)
    else:
        seeds_file.write_text("2000\n\n 2001\n2002 \n\n")

    assert seeds_from_extern(seeds_file) == [2000, 2001, 2002]


def test_that_seeds_from_extern_ignores_leading_empty_rows_and_columns(tmp_path):
    seeds_file = tmp_path / "seeds.xlsx"
    with xlsxwriter.Workbook(seeds_file) as workbook:
        worksheet = workbook.add_worksheet()
        worksheet.write_number(2, 1, 2000)
        worksheet.write_number(3, 1, 2001)
        worksheet.write_number(4, 1, 2002)

    assert seeds_from_extern(seeds_file) == [2000, 2001, 2002]


def test_that_non_integer_seed_values_raise_value_error(tmp_path):
    seeds_file = tmp_path / "seeds.txt"
    seeds_file.write_text("2000\n2000.5\n")

    with pytest.raises(ValueError, match=r"2000\.5"):
        seeds_from_extern(seeds_file)


def test_that_seeds_from_extern_rejects_unsupported_file_extensions():
    with pytest.raises(ValueError, match=r"end with \.xlsx \.csv or \.txt"):
        seeds_from_extern("seeds.json")


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
            pl.DataFrame({"SOURCE": from_values[:1]}), dependencies=dependencies
        )
