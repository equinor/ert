from textwrap import dedent

import pytest

from ert._c_wrappers.enkf.ert_config_lint import lint_file
from ert._c_wrappers.enkf.ert_config_lint_header import LintLocation, LintType

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
