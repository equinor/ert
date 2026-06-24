from pathlib import Path

import polars as pl
import pytest

from ert.utils import assert_schema, makedirs_if_needed


def test_makedirs(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # assert output dir (/tmp/tmpXXXX) is empty
    assert not output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 0

    # create output folder
    makedirs_if_needed(output_dir)

    # assert output folder created
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1


def test_makedirs_already_exists(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1

    # run makedirs_if_needed again, verify nothing happened
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1


def test_makedirs_roll_existing(change_to_tmpdir):
    output_dir = Path("unittest_everest_output")
    cwd = Path.cwd()

    # create outputfolder and verify it's existing
    makedirs_if_needed(output_dir)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 1

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, roll_if_exists=True)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 2

    # run makedirs_if_needed again, verify old dir rolled
    makedirs_if_needed(output_dir, roll_if_exists=True)
    assert output_dir.is_dir()
    assert len(list(cwd.iterdir())) == 3


def test_that_assert_schema_raises_when_schema_datatype_is_not_as_expected():
    schema = {"col": pl.Float32}
    df1 = pl.DataFrame({"col": pl.Series([1.11, 2.22], dtype=pl.Float32)})
    df2 = pl.DataFrame({"col": [1.11, 2.22]})

    assert_schema(df1, schema)
    with pytest.raises(AssertionError, match="Expected schema"):
        assert_schema(df2, schema)


def test_that_assert_schema_raises_when_schema_column_order_is_not_as_expected():
    schema = {"col1": pl.String, "col2": pl.String}
    df1 = pl.DataFrame(
        {
            "col1": pl.Series(["a", "b"], dtype=pl.String),
            "col2": pl.Series(["c", "d"], dtype=pl.String),
        }
    )
    df2 = pl.DataFrame(
        {
            "col2": pl.Series(["c", "d"], dtype=pl.String),
            "col1": pl.Series(["a", "b"], dtype=pl.String),
        }
    )

    assert_schema(df1, schema)
    with pytest.raises(AssertionError, match="Expected schema"):
        assert_schema(df2, schema)
    assert_schema(df2, schema, check_column_order=False)
