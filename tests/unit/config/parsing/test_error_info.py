import hypothesis.strategies as st
from hypothesis import given

from ert.config.parsing import ErrorInfo

error_infos = st.builds(ErrorInfo)


@given(error_infos, error_infos)
def test_that_gt_is_consistent_with_eq(a, b):
    if a > b or a < b:
        assert a != b
    else:
        assert a == b


def test_that_sort_orders_by_filename_first():
    assert sorted(
        [
            ErrorInfo("1", filename="a"),
            ErrorInfo("1", filename="b"),
            ErrorInfo("2", filename="a"),
            ErrorInfo("2", filename="b"),
        ]
    ) == [
        ErrorInfo("1", filename="a"),
        ErrorInfo("2", filename="a"),
        ErrorInfo("1", filename="b"),
        ErrorInfo("2", filename="b"),
    ]


def test_that_sort_orders_by_line_number_after_filename():
    assert sorted(
        [
            ErrorInfo("", filename="a", line=1),
            ErrorInfo("", filename="b", line=1),
            ErrorInfo("", filename="a", line=2),
            ErrorInfo("", filename="b", line=2),
        ]
    ) == [
        ErrorInfo("", filename="a", line=1),
        ErrorInfo("", filename="a", line=2),
        ErrorInfo("", filename="b", line=1),
        ErrorInfo("", filename="b", line=2),
    ]


@given(st.text())
def test_str_without_location_is_just_message(message):
    assert str(ErrorInfo(message)) == message


@given(st.text(), st.text())
def test_str_with_filename_concatenates_with_colon(message, filename):
    assert str(ErrorInfo(message, filename=filename)) == filename + ": " + message


@given(st.text(), st.text(), st.integers())
def test_str_with_filename_and_line_formats(message, filename, line):
    assert (
        str(ErrorInfo(message, filename=filename, line=line))
        == filename + ": Line " + str(line) + " " + message
    )


@given(st.text(), st.text(), st.integers(), st.integers())
def test_str_with_just_start_column_formats(message, filename, line, start_column):
    assert (
        str(
            ErrorInfo(
                message,
                filename=filename,
                line=line,
                column=start_column,
            )
        )
        == f"{filename}: Line {line} (Column {start_column}): " + message
    )


@given(st.text(), st.text(), st.integers(), st.integers(), st.integers())
def test_str_with_column_formats(message, filename, line, start_column, end_column):
    assert (
        str(
            ErrorInfo(
                message,
                filename=filename,
                line=line,
                column=start_column,
                end_column=end_column,
            )
        )
        == f"{filename}: Line {line} (Column {start_column}-{end_column}): " + message
    )
