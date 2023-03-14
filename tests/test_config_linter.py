from textwrap import dedent

import pytest

from ert._c_wrappers.enkf.ert_config_lint import lint_file
from ert._c_wrappers.enkf.ert_config_lint_header import LintLocation, LintType

# List of errors (from ert_config.py)
# 1. Negative MAX_RUNNING value for QUEUE_OPTION
# 2. SUMMARY keyword used without specifying ECLBASE
# 3. Loading GEN_KW from files created by forward model not supported
# 4. Loading GEN_KW from files requires %d in file format
# 5. Job not in list of installed jobs
# 6. Forwarded error from ValueError parsing args (explore a bit more)
# 7. Run mode not supported for Hook workflow
# 8. Cannot set up hook for non-existing job name
# 9. Unable to locate job directory

# (from lark_parser.py)
# 10. kw argument must be one of { ... } but was
# 11. kw must have boolean value
# 12. kw must be int
# 13. kw must be number
# 14. file or directory not found
# 15. cannot find executable
# 16. file is not executable
# 17. kw takes at most N arguments
# 18. e.errors\n was used in token.value at line token.line
# 19. kw must be set
# 20. cannot define kw to value
# 21. cannot define data_kw to value
# 22. unrecognized kw
# 23. could not read kw value for key
# 24. cannot mix arglist with paranthesis and without in {node}
# 25. could not read argument val
# 26. kw needs at least argcmin args
# 27. kw takes max argcmax args
# 28. unexpected top level statement
# 29. include must have exactly one arg
# 30. include kw must be given filepath
# 31. cyclical import detected, {file} already included
# 32. INCLUDE: file not found
# 33. DEFINE/DATA_KW must have two or more args with given format, parser said
#

# List of warnings (from ert_config.py)
# 10.Can not use JOBNAME and ECLBASE at same time, ECLBASE is ignored
# 11. ExtJobInvalidArgsException as err: (whatever comes out of that)
# 12. Environment variable is skipped due to unmatched define
# 13. Loading workflow job failed with error
# 14. Unable to open job directory
# 15. ErtScriptLoadFailure
# 16. Workflow added twice
# 17. Duplicate forward model job
# 18. No files found in job directory
# 19. Duplicate forward model, choosing this

# Test approach for lints
# 1. Start with an OK (in order config file)
# 2. At varying lines, insert errors, then check if they are linted
# 3. First one by one, then in combination

# For includes, test insert below/above DIRECTLY installed jobs(?)
# test for indirectly?


poly_content = """
JOBNAME poly_%d

QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 50

RUNPATH poly_out/realization-<IENS>/iter-<ITER>

OBS_CONFIG observations
TIME_MAP time_map

NUM_REALIZATIONS 100
MIN_REALIZATIONS 1

GEN_KW COEFFS coeff.tmpl coeffs.json coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII

INSTALL_JOB poly_eval POLY_EVAL
SIMULATION_JOB poly_eval
"""

poly_files = {
    "observations": "",
    "time_map": "",
    "coeffs.json": "",
    "coeff_priors": "",
    "coeff.tmpl": "",
    "POLY_EVAL": "EXECUTABLE job_dispatch.py",
}


def create_config_file(content: str, file_name="test.ert"):
    test_config_contents = dedent(content)
    with open(file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)


def string_with_added_line(
    original_string: str, new_line_content: str, new_line_index: int
) -> str:
    lines = original_string.splitlines()
    lines.insert(new_line_index, new_line_content)
    return "\n".join(lines)


def create_dummy_files(files):
    for name, content in files.items():
        with open(name, "w+", encoding="utf8") as f:
            f.write(content)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skip(reason="Not yet implemented")
def test_that_poly_example_raises_no_errors():
    create_config_file(file_name="test.ert", content=poly_content)
    create_dummy_files(poly_files)
    linter = lint_file("test.ert")

    assert linter.is_empty()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skip(reason="Not yet implemented")
def test_poly_example_with_non_existing_include():
    pytest.skip()
    num_lines = poly_content.count("\n")

    for i in range(0, num_lines):
        modified_content = string_with_added_line(
            original_string=poly_content,
            new_line_content="INCLUDE NON_EXISTING.txt",
            new_line_index=i,
        )

        create_config_file(file_name="test.ert", content=modified_content)
        create_dummy_files(poly_files)
        linter = lint_file("test.ert")

        assert linter.number_of_lints() == 1

        the_lint = linter.get(0)

        assert the_lint.lint_type == LintType.ERROR
        assert the_lint.location == LintLocation(
            line=(i + 1), end_line=(i + 1), column=1, end_column=25
        )
