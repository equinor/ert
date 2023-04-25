import os
import re
import stat
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pytest
from hypothesis import given, strategies

from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError
from ert.parsing.error_info import ErrorInfo

test_config_file_base = "test"
test_config_filename = f"{test_config_file_base}.ert"


@dataclass
class FileDetail:
    contents: str
    is_executable: bool = False


@dataclass
class ExpectedErrorInfo:
    filename: str = "test.ert"
    line: Optional[int] = None
    column: Optional[int] = None
    end_column: Optional[int] = None
    other_files: Optional[Dict[str, FileDetail]] = None
    match: Optional[str] = None
    count: Optional[int] = None


def write_files(files: Optional[Dict[str, Union[str, FileDetail]]] = None):
    if files is not None:
        for other_filename, content in files.items():
            with open(other_filename, mode="w", encoding="utf-8") as fh:
                if isinstance(content, FileDetail):
                    fh.writelines(content.contents)

                    if not content.is_executable:
                        os.chmod(other_filename, stat.S_IREAD)
                else:
                    fh.write(content)


def find_and_assert_errors_matching_filename(
    errors: List[ErrorInfo], filename: Optional[str]
):
    matching_errors = (
        [err for err in errors if filename in err.filename]
        if filename is not None
        else errors
    )

    assert len(matching_errors) > 0, (
        f"Expected minimum 1 error matching filename" f" {filename}, got 0"
    )

    return matching_errors


def find_and_assert_errors_matching_location(
    errors: List[ErrorInfo],
    line: Optional[int] = None,
    column: Optional[int] = None,
    end_column: Optional[int] = None,
):
    def equals_or_expected_any(actual, expected):
        return True if expected is None else actual == expected

    matching_errors = [
        x
        for x in errors
        if equals_or_expected_any(x.line, line)
        and equals_or_expected_any(x.column, column)
        and equals_or_expected_any(x.end_column, end_column)
    ]

    def none_to_star(val: Optional[int] = None):
        return "*" if val is None else val

    assert len(matching_errors) > 0, (
        "Expected to find at least 1 error matching location"
        f"(line={none_to_star(line)},"
        f"column={none_to_star(column)},"
        f"end_column{none_to_star(end_column)})"
    )

    return matching_errors


def find_and_assert_errors_matching_message(
    errors: List[ErrorInfo], match: Optional[str] = None
):
    if match is None:
        return errors

    re_match = re.compile(match)
    matching_errors = [err for err in errors if re.search(re_match, err.message)]

    assert len(matching_errors) > 0, f"Expected to find error matching message {match}"

    return matching_errors


def assert_that_config_leads_to_error(
    config_file_contents: str,
    expected_error: ExpectedErrorInfo,
    config_filename: str = "test.ert",
):
    write_files(
        {config_filename: config_file_contents, **(expected_error.other_files or {})}
    )

    with pytest.raises(ConfigValidationError) as caught_error:
        ErtConfig.from_file(config_filename, use_new_parser=True)

    collected_errors = caught_error.value.errors

    # Find errors in matching file
    errors_matching_filename = find_and_assert_errors_matching_filename(
        errors=collected_errors, filename=expected_error.filename
    )

    errors_matching_location = find_and_assert_errors_matching_location(
        errors=errors_matching_filename,
        line=expected_error.line,
        column=expected_error.column,
        end_column=expected_error.end_column,
    )

    errors_matching_message = find_and_assert_errors_matching_message(
        errors=errors_matching_location, match=expected_error.match
    )

    if expected_error.count is not None:
        assert len(errors_matching_message) == expected_error.count, (
            f"Expected to find exactly {expected_error.count} errors, "
            f"found {len(errors_matching_message)}."
        )


def assert_that_config_does_not_lead_to_error(
    config_file_contents: str,
    unexpected_error: ExpectedErrorInfo,
    config_filename: str = "test.ert",
):
    with pytest.raises(AssertionError):
        assert_that_config_leads_to_error(
            config_file_contents,
            expected_error=unexpected_error,
            config_filename=config_filename,
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_disallowed_argument_is_located_1fn():
    assert_that_config_leads_to_error(
        config_file_contents="""
QUEUE_OPTION DOCAL MAX_RUNNING 4
""",
        expected_error=ExpectedErrorInfo(
            line=2,
            column=14,
            end_column=19,
            match="argument .* must be one of",
        ),
    )


@pytest.mark.parametrize(
    "contents, expected_errors",
    [
        (
            """
QUEUE_OPTION DOCAL MAX_RUNNING 4
STOP_LONG_RUNNING flase
NUM_REALIZATIONS not_int
ENKF_ALPHA not_float
RUN_TEMPLATE dsajldkald/sdjkahsjka/wqehwqhdsa
JOB_SCRIPT dnsjklajdlksaljd/dhs7sh/qhwhe
JOB_SCRIPT non_executable_file
NUM_REALIZATIONS 1 2 3 4 5
NUM_REALIZATIONS
""",
            [
                ExpectedErrorInfo(
                    line=2,
                    column=14,
                    end_column=19,
                    match="argument .* must be one of",
                ),
                ExpectedErrorInfo(
                    line=3,
                    column=19,
                    end_column=24,
                    match="must have a boolean value",
                ),
                ExpectedErrorInfo(
                    line=4,
                    column=18,
                    end_column=25,
                    match="must have an integer value",
                ),
                ExpectedErrorInfo(
                    line=5,
                    column=12,
                    end_column=21,
                    match="must have a number",
                ),
                ExpectedErrorInfo(
                    line=6,
                    column=14,
                    end_column=46,
                    match="Cannot find file or directory",
                ),
                ExpectedErrorInfo(
                    line=7,
                    column=12,
                    end_column=41,
                    match="Could not find executable",
                ),
                ExpectedErrorInfo(
                    other_files={
                        "non_executable_file": FileDetail(
                            contents="", is_executable=False
                        )
                    },
                    line=8,
                    column=12,
                    end_column=31,
                    match="File not executable",
                ),
                ExpectedErrorInfo(
                    line=9,
                    column=1,
                    end_column=17,
                    match="must have maximum",
                ),
                ExpectedErrorInfo(
                    line=10,
                    column=1,
                    end_column=17,
                    match="must have at least",
                ),
            ],
        )
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_multiple_keyword_specific_tokens_are_located(contents, expected_errors):
    for expected_error in expected_errors:
        assert_that_config_leads_to_error(
            config_file_contents=contents, expected_error=expected_error
        )


@pytest.mark.usefixtures("use_tmpdir")
@given(
    strategies.lists(
        strategies.sampled_from(
            [
                ("QUEUE_OPTION DOCAL MAX_RUNNING 4", {"column": 14, "end_column": 19}),
                ("STOP_LONG_RUNNING flase", {"column": 19, "end_column": 24}),
                ("NUM_REALIZATIONS not_int", {"column": 18, "end_column": 25}),
                ("ENKF_ALPHA not_float", {"column": 12, "end_column": 21}),
                (
                    "RUN_TEMPLATE dsajldkald/sdjkahsjka/wqehwqhdsa",
                    {"column": 14, "end_column": 46},
                ),
                (
                    "JOB_SCRIPT dnsjklajdlksaljd/dhs7sh/qhwhe",
                    {"column": 12, "end_column": 41},
                ),
                ("NUM_REALIZATIONS 1 2 3 4 5", {"column": 1, "end_column": 17}),
                ("NUM_REALIZATIONS", {"column": 1, "end_column": 17}),
            ]
        ),
        min_size=1,
        max_size=10,
    )
)
def test_that_multiple_keyword_specific_tokens_are_located_shuffle(error_lines):
    contents = "\n".join([line[0] for line in error_lines])
    expected_errors = [
        ExpectedErrorInfo(
            line=(i + 1),
            column=line[1]["column"],
            end_column=line[1]["end_column"],
        )
        for i, line in enumerate(error_lines)
    ]

    for err in expected_errors:
        assert_that_config_leads_to_error(
            config_file_contents=contents, expected_error=err
        )


def test_not_declared_num_realizations_leads_to_only_one_error():
    assert_that_config_leads_to_error(
        config_file_contents="",
        expected_error=ExpectedErrorInfo(match=".* must be set", count=1),
    )


def test_invalid_num_realizations_does_not_lead_to_unset_error():
    assert_that_config_does_not_lead_to_error(
        config_file_contents="NUM_REALIZATIONS ert",
        unexpected_error=ExpectedErrorInfo(match=".* must be set", count=1),
    )

    assert_that_config_leads_to_error(
        config_file_contents="NUM_REALIZATIONS ert",
        expected_error=ExpectedErrorInfo(
            line=1,
            column=18,
            end_column=21,
            match="must have an integer value",
        ),
    )
