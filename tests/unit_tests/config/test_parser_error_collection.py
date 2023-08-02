# pylint: disable=C0302
import os
import re
import stat
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional, Union, cast

import pytest
from hypothesis import given, strategies

from ert.config import ConfigValidationError, ErtConfig
from ert.config.parsing import ErrorInfo

test_config_file_base = "test"
test_config_filename = f"{test_config_file_base}.ert"


@dataclass
class FileDetail:
    contents: str
    is_executable: bool = False
    is_readable: bool = True


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

                    if not content.is_readable:
                        os.chmod(other_filename, ~0o400)
                else:
                    fh.write(content)


def find_and_assert_errors_matching_filename(
    errors: List[ErrorInfo], filename: Optional[str]
):
    matching_errors = (
        [err for err in errors if err.filename is not None and filename in err.filename]
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
        f"end_column={none_to_star(end_column)})"
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
        ErtConfig.from_file(config_filename)
        # If the ert config did not raise any errors
        # we manually raise an "empty" error to make
        # this raise an assertion error that can be
        # acted upon from assert_that_config_does_not_lead_to_error
        raise ConfigValidationError(errors=[])

    collected_errors = caught_error.value.errors

    if len(collected_errors) == 0:
        raise AssertionError("Config did not lead to any errors")

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


def assert_that_config_leads_to_warning(
    config_file_contents: str,
    expected_error: ExpectedErrorInfo,
    config_filename: str = "test.ert",
):
    write_files(
        {config_filename: config_file_contents, **(expected_error.other_files or {})}
    )

    ert_config = ErtConfig.from_file(config_filename)

    warnings_matching_filename = find_and_assert_errors_matching_filename(
        errors=cast(List[ErrorInfo], ert_config.warning_infos),
        filename=expected_error.filename,
    )

    warnings_matching_location = find_and_assert_errors_matching_location(
        errors=warnings_matching_filename,
        line=expected_error.line,
        column=expected_error.column,
        end_column=expected_error.end_column,
    )

    warnings_matching_message = find_and_assert_errors_matching_message(
        errors=warnings_matching_location, match=expected_error.match
    )

    if expected_error.count is not None:
        assert len(warnings_matching_message) == expected_error.count, (
            f"Expected to find exactly {expected_error.count} errors, "
            f"found {len(warnings_matching_message)}."
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
        config_file_contents="QUEUE_OPTION DOCAL MAX_RUNNING 4",
        expected_error=ExpectedErrorInfo(
            line=1,
            column=14,
            end_column=19,
            match="argument .* must be one of",
        ),
    )


@pytest.mark.parametrize(
    "contents, expected_errors",
    [
        (
            dedent(
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
                """
            ),
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


@pytest.mark.usefixtures("use_tmpdir")
def test_not_declared_num_realizations_leads_to_only_one_error():
    assert_that_config_leads_to_error(
        config_file_contents="",
        expected_error=ExpectedErrorInfo(match=".* must be set", count=1),
    )


@pytest.mark.usefixtures("use_tmpdir")
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


@pytest.mark.usefixtures("use_tmpdir")
def test_summary_without_eclbase():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            NUM_REALIZATIONS 1
            SUMMARY summary
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=3,
            column=1,
            end_column=8,
            match="When using SUMMARY keyword, the config must also specify ECLBASE",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_include_non_existing_file_errors_with_location(tmpdir):
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            JOBNAME my_name%d
            NUM_REALIZATIONS 1
            INCLUDE does_not_exists
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=4,
            column=9,
            end_column=24,
            match=r"INCLUDE file:.*does_not_exists not found",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_include_with_too_many_args_errors_with_location(tmpdir):
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            JOBNAME my_name%d
            INCLUDE something arg1 arg2 dot dotdot argn
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=3,
            column=19,
            end_column=44,
            match=r"Keyword:INCLUDE must have exactly one argument",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_include_with_too_many_args_error_is_located_indirect(tmpdir):
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            JOBNAME my_name%d
            INCLUDE something.ert
            """
        ),
        expected_error=ExpectedErrorInfo(
            filename="something.ert",
            line=1,
            column=14,
            end_column=39,
            match=r"Keyword:INCLUDE must have exactly one argument",
            other_files={
                "something.ert": FileDetail(
                    contents="INCLUDE arg1 arg2 arg3 dot dotdot argN"
                )
            },
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_queue_option_max_running_non_int():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            NUM_REALIZATIONS 1
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING ert
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=4,
            column=32,
            end_column=35,
            match="not an integer",
        ),
    )


@pytest.mark.parametrize(
    "contents, expected_errors",
    [
        (
            dedent(
                """
                INCLUDE does_not_exist0
                NUM_REALIZATIONS 1
                INCLUDE does_not_exist1
                INCLUDE does_not_exist2
                INCLUDE does_not_exist3
                JOBNAME my_name%d
                INCLUDE does_not_exist4
                INCLUDE does_not_exist5
                INCLUDE does_not_exist6
                INCLUDE does_not_exist7
                """
            ),
            [
                *[
                    ExpectedErrorInfo(
                        line=line,
                        column=9,
                        end_column=24,
                        match=f"INCLUDE file:.*does_not_exist{i} not found",
                    )
                    for i, line in enumerate([2, 4, 5, 6, 8, 9, 10, 11])
                ],
            ],
        )
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_multiple_include_non_existing_files_are_located(contents, expected_errors):
    for expected_error in expected_errors:
        assert_that_config_leads_to_error(
            config_file_contents=contents, expected_error=expected_error
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_cyclical_import_error_is_located():
    assert_that_config_leads_to_error(
        config_file_contents="""
NUM_REALIZATIONS  1
INCLUDE include.ert
""",
        expected_error=ExpectedErrorInfo(
            other_files={
                "include.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE test.ert
                        """
                    )
                )
            },
            line=3,
            column=9,
            end_column=20,
            filename="test.ert",
            match="Cyclical import detected, test.ert->include.ert->test.ert",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_cyclical_import_error_is_located_branch():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            NUM_REALIZATIONS  1
            INCLUDE include1.ert
            """
        ),
        expected_error=ExpectedErrorInfo(
            other_files={
                "include1.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE include2.ert
                        """
                    )
                ),
                "include2.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE include3.ert
                        """
                    )
                ),
                "include3.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE include4.ert
                        """
                    )
                ),
                "include4.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE include5.ert
                        """
                    )
                ),
                "include5.ert": FileDetail(
                    contents=dedent(
                        """
                        JOBNAME included
                        INCLUDE test.ert
                        """
                    )
                ),
            },
            line=3,
            column=9,
            end_column=21,
            filename="test.ert",
            match="Cyclical import detected, "
            "test.ert->include1.ert->include2.ert->include3.ert"
            "->include4.ert->include5.ert->test.ert",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("n", range(1, 10))
def test_that_cyclical_import_error_is_located_stop_early(n):
    other_include_files = {
        f"include{i}.ert": FileDetail(
            contents=dedent(
                f"""
                JOBNAME include{i}
                INCLUDE include{i+1}.ert
                """
            )
        )
        for i in range(n)
    }

    expected_match = (
        f"test.ert->{'->'.join([f'include{i}.ert' for i in range(n+1)])}->test.ert"
    )

    expected_error = ExpectedErrorInfo(
        other_files={
            **other_include_files,
            f"include{n}.ert": FileDetail(
                contents=dedent(
                    f"""
                    JOBNAME include{n+1}
                    INCLUDE test.ert
                    """
                )
            ),
        },
        line=3,
        column=9,
        end_column=21,
        filename="test.ert",
        match=expected_match,
    )

    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            NUM_REALIZATIONS 1
            INCLUDE include0.ert
            """
        ),
        expected_error=expected_error,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_queue_option_max_running_negative():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
            NUM_REALIZATIONS 1
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING -1
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=4,
            column=32,
            end_column=34,
            match="negative",
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unicode_decode_error_is_localized_first_line():
    with open("test.ert", "ab") as f:
        f.write(b"\xff")
        f.write(
            bytes(
                dedent(
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
                """
                ),
                "utf-8",
            )
        )

    with pytest.raises(
        ConfigValidationError,
        match="Unsupported non UTF-8 character 'ÿ' found in file: .*test.ert",
    ) as caught_error:
        ErtConfig.from_file("test.ert")

    collected_errors = caught_error.value.errors

    # Expect parsing to stop from this invalid character
    assert len(collected_errors) == 1
    assert collected_errors[0].line == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unicode_decode_error_is_localized_random_line_single_insert():
    lines = """
        QUEUE_OPTION DOCAL MAX_RUNNING 4
        STOP_LONG_RUNNING flase
        NUM_REALIZATIONS not_int
        ENKF_ALPHA not_float
        RUN_TEMPLATE dsajldkald/sdjkahsjka/wqehwqhdsa
        JOB_SCRIPT dnsjklajdlksaljd/dhs7sh/qhwhe
        JOB_SCRIPT non_executable_file
        NUM_REALIZATIONS 1 2 3 4 5
        NUM_REALIZATIONS
    """.splitlines()

    for insertion_index in range(1, len(lines)):
        before = lines[0:insertion_index]
        after = lines[insertion_index : len(lines)]

        with open("test.ert", "w", encoding="utf-8") as f:
            f.write("\n".join(before) + "\n")

        with open("test.ert", "ab") as f:
            f.write(b"\xff")

        with open("test.ert", "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(after))

        with pytest.raises(
            ConfigValidationError,
            match="Unsupported non UTF-8 character " "'ÿ' found in file: .*test.ert",
        ) as caught_error:
            ErtConfig.from_file("test.ert")

        collected_errors = caught_error.value.errors

        # Expect parsing to stop from this invalid character
        assert len(collected_errors) == 1

        assert collected_errors[0].line == insertion_index + 1
        assert collected_errors[0].end_line == insertion_index + 1
        assert collected_errors[0].column == 0
        assert collected_errors[0].end_column == -1


@pytest.mark.usefixtures("use_tmpdir")
@given(
    lines=strategies.lists(
        strategies.sampled_from(
            [
                "QUEUE_OPTION DOCAL MAX_RUNNING 4",
                "STOP_LONG_RUNNING flase",
                "NUM_REALIZATIONS not_int",
                "ENKF_ALPHA not_float",
                "RUN_TEMPLATE dsajldkald/sdjkahsjka/wqehwqhdsa",
                "JOB_SCRIPT dnsjklajdlksaljd/dhs7sh/qhwhe",
                "JOB_SCRIPT non_executable_file",
                "NUM_REALIZATIONS 1 2 3 4 5",
                "NUM_REALIZATIONS",
            ]
        ),
        min_size=5,
        max_size=10,
    ),
    insertion_indices=strategies.lists(
        strategies.sampled_from(range(4)), min_size=2, max_size=5
    ),
)
def test_that_unicode_decode_error_is_localized_multiple_random_inserts(
    lines, insertion_indices
):
    write_infos = [{"type": "utf-8", "content": x} for x in lines]

    for offset, index in enumerate(sorted(insertion_indices)):
        write_infos.insert(index + offset, {"type": "bytes", "content": b"\xff"})

    for i, info in enumerate(write_infos):
        if info["type"] == "utf-8":
            with open("test.ert", "w" if i == 0 else "a", encoding="utf-8") as f:
                f.write(info["content"])
        elif info["type"] == "bytes":
            with open("test.ert", "wb" if i == 0 else "ab") as f:
                f.write(info["content"])

        if i < (len(write_infos) - 1):
            with open("test.ert", "a", encoding="utf-8") as f:
                f.write("\n")

    with pytest.raises(
        ConfigValidationError,
        match="Unsupported non UTF-8 character 'ÿ' found in file: .*test.ert",
    ) as caught_error:
        ErtConfig.from_file("test.ert")

    collected_errors = caught_error.value.errors

    # Expect parsing to stop from this invalid character
    assert len(collected_errors) == len(insertion_indices)

    for line, _ in [(i, x) for i, x in enumerate(write_infos) if x["type"] == "bytes"]:
        # Expect there to be an error on the line
        expected_error = next(x for x in collected_errors if x.line == line + 1)
        assert expected_error is not None, f"Expected to find error on line {line + 1}"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_existing_workflow_is_localized():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
                NUM_REALIZATIONS  1
                LOAD_WORKFLOW does_not_exist
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=3,
            column=15,
            end_column=29,
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_readable_workflow_job_is_localized():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
                NUM_REALIZATIONS  1
LOAD_WORKFLOW_JOB exists_but_not_runnable
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=3,
            column=19,
            end_column=42,
            filename="test.ert",
            other_files={
                "exists_but_not_runnable": FileDetail(is_readable=False, contents="")
            },
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_readable_workflow_job_in_directory_is_localized():
    os.mkdir("hello")
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
WORKFLOW_JOB_DIRECTORY hello
            """
        ),
        expected_error=ExpectedErrorInfo(
            filename="test.ert",
            line=3,
            column=24,
            end_column=29,
            other_files={
                "hello/exists_but_not_runnable": FileDetail(
                    is_readable=False, contents=""
                )
            },
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_hook_workflow_without_existing_job_error_is_located():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
HOOK_WORKFLOW NO_SUCH_JOB POST_SIMULATION
            """
        ),
        expected_error=ExpectedErrorInfo(
            line=3,
            column=15,
            end_column=26,
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_int_followed_by_negative_wont_re_trigger_negative_error():
    assert_that_config_does_not_lead_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
QUEUE_OPTION LOCAL MAX_RUNNING -1
QUEUE_OPTION LOCAL MAX_RUNNING ert
            """
        ),
        unexpected_error=ExpectedErrorInfo(
            match="is negative: 'ert'",
        ),
    )


@pytest.mark.parametrize("dirname", ["the_dir", "/tmp"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_executable_directory_errors(dirname):
    os.mkdir("the_dir")
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            f"""
NUM_REALIZATIONS  1
JOB_SCRIPT {dirname}

            """
        ),
        expected_error=ExpectedErrorInfo(
            match="is a directory",
            line=3,
            column=12,
            end_column=12 + len(dirname),
        ),
    )


@pytest.mark.parametrize(
    "contents, expected_errors",
    [
        (
            """
NUM_REALIZATIONS  1
RUNPATH %d

RELPERM
MULTZ Hehe 2 l 1 2
EQUIL Hehe 2 l 1 2
GEN_PARAM Hehe 2 l 1 2
MULTFLT Hehe 2 l 1 2

UMASK 1 2 3 4 111
LOG_FILE 1 2 3 4 111
LOG_LEVEL 1 2 3 4 111
ENKF_RERUN 1 2 3 4 111

RSH_HOST boq
RSH_COMMAND qqqwe
MAX_RUNNING_RSH assadsa 3 12w qwsa

LSF_SERVER
LSF_QUEUE
MAX_RUNNING_LSF
MAX_RUNNING_LOCAL

SCHEDULE_PREDICTION_FILE
HAVANA_FAULT
REFCASE_LIST
RFTPATH
END_DATE
CASE_TABLE
RERUN_START
DELETE_RUNPATH
PLOT_SETTINGS
UPDATE_PATH A B
UPDATE_SETTINGS A

DEFINE A <2>
""",
            [
                ExpectedErrorInfo(
                    line=3,
                    column=1,
                    end_column=8,
                    match="RUNPATH keyword contains deprecated value placeholders:",
                ),
                *[
                    ExpectedErrorInfo(
                        line=5 + i,
                        column=1,
                        end_column=1 + len(kw),
                        match=f"{kw} .* replaced by the GEN_KW",
                    )
                    for i, kw in enumerate(
                        ["RELPERM", "MULTZ", "EQUIL", "GEN_PARAM", "MULTFLT"]
                    )
                ],
                *[
                    ExpectedErrorInfo(
                        line=11 + i,
                        column=1,
                        end_column=1 + len(kw),
                        match=f"The keyword {kw} no longer has any effect",
                    )
                    for i, kw in enumerate(
                        [
                            "UMASK",
                            "LOG_FILE",
                            "LOG_LEVEL",
                            "ENKF_RERUN",
                        ]
                    )
                ],
                *[
                    ExpectedErrorInfo(
                        line=16 + i,
                        column=1,
                        end_column=1 + len(kw),
                        match=f"The {kw} was used for the deprecated "
                        "and removed support for RSH queues",
                    )
                    for i, kw in enumerate(
                        ["RSH_HOST", "RSH_COMMAND", "MAX_RUNNING_RSH"]
                    )
                ],
                *[
                    ExpectedErrorInfo(
                        line=20 + i,
                        column=1,
                        end_column=1 + len(kw),
                        match=f"The {kw} keyword has been removed",
                    )
                    for i, kw in enumerate(
                        [
                            "LSF_SERVER",
                            "LSF_QUEUE",
                            "MAX_RUNNING_LSF",
                            "MAX_RUNNING_LOCAL",
                        ]
                    )
                ],
                *[
                    ExpectedErrorInfo(
                        line=25 + i,
                        column=1,
                        end_column=1 + len(kw),
                        match=f"{kw}",
                    )
                    for i, kw in enumerate(
                        [
                            "SCHEDULE_PREDICTION_FILE",
                            "HAVANA_FAULT",
                            "REFCASE_LIST",
                            "RFTPATH",
                            "END_DATE",
                            "CASE_TABLE",
                            "RERUN_START",
                            "DELETE_RUNPATH",
                            "PLOT_SETTINGS",
                            "UPDATE_PATH",
                            "UPDATE_SETTINGS",
                        ]
                    )
                ],
                ExpectedErrorInfo(
                    line=34,
                    column=1,
                    end_column=12,
                    match="UPDATE_PATH keyword has been removed",
                ),
            ],
        )
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_deprecations_are_handled(contents, expected_errors):
    for expected_error in expected_errors:
        assert_that_config_leads_to_warning(
            config_file_contents=contents, expected_error=expected_error
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_invalid_ensemble_result_file_errors():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
GEN_DATA RFT_3-1_R_DATA INPUT_FORMAT:ASCII REPORT_STEPS:100 RESULT_FILE:RFT_3-1_R_<ITER>

            """
        ),
        expected_error=ExpectedErrorInfo(
            match="must have an embedded %d",
            line=3,
            column=61,
            end_column=89,
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_missing_report_steps_errors():
    assert_that_config_leads_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
GEN_DATA RFT_3-1_R_DATA INPUT_FORMAT:ASCII RESULT_FILE:RFT_3-1_R%d

            """
        ),
        expected_error=ExpectedErrorInfo(
            match="REPORT_STEPS",
            line=3,
            column=1,
            end_column=9,
        ),
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_valid_gen_data_does_not_error():
    assert_that_config_does_not_lead_to_error(
        config_file_contents=dedent(
            """
NUM_REALIZATIONS  1
GEN_DATA RFT_3-1_R_DATA INPUT_FORMAT:ASCII REPORT_STEPS:100 RESULT_FILE:RFT_3-1_R%d

            """
        ),
        unexpected_error=ExpectedErrorInfo(),
    )
