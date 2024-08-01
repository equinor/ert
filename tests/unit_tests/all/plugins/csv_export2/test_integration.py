import os
import shutil
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import hypothesis.strategies as st
import pandas as pd
import pytest
import rstcheck_core.checker
from hypothesis import given, settings

from ert.plugins import ErtPluginContext, ErtPluginManager
from ert.plugins.hook_implementations.workflows.csv_export2 import csv_export2
from tests.integration_tests.run_cli import run_cli_with_pm
from tests.unit_tests.config.summary_generator import summaries

KEYWORDS = ["FGPT", "FLPT", "FOPT", "FVPT", "FWPT"]

monthly_summary = summaries(
    start_date=st.just(datetime(year=1976, month=1, day=1)),
    summary_keys=st.just(KEYWORDS),
    time_deltas=st.just([31, 60, 91]),
    use_days=st.just(True),
    report_step_only=True,
)

yearly_summary = summaries(
    start_date=st.just(datetime(year=1976, month=1, day=1)),
    summary_keys=st.just(KEYWORDS),
    time_deltas=st.just([365]),
    use_days=st.just(True),
    report_step_only=True,
)


def mock_data(summary, reals, iters, parameters=True):
    # pylint: disable=consider-using-f-string
    """From a single UNSMRY file, produce arbitrary sized ensembles.

    Summary data will be equivalent over realizations, but the
    parameters.txt is made unique.

    Writes realization-*/iter-* file structure in cwd.

    Args:
        reals (list): integers with realization indices wanted
        iters (list): integers with iter indices wanted
        parameters (bool): Whether to write parameters.txt in each runpath
    """
    smspec, unsmry = summary
    for real in reals:
        for iteration in iters:
            runpath = os.path.join(f"realization-{real}", f"iter-{iteration}")

            os.makedirs(runpath, exist_ok=True)

            unsmry.to_file(os.path.join(runpath, f"RES_{real}.UNSMRY"))
            smspec.to_file(
                os.path.join(runpath, f"RES_{real}.SMSPEC"),
            )
            if parameters:
                with open(
                    os.path.join(runpath, "parameters.txt"), "w", encoding="utf-8"
                ) as p_fileh:
                    p_fileh.write(f"FOO 1{real}{iteration}")
            # Ensure fmu-ensemble does not complain on missing STATUS
            with open(os.path.join(runpath, "STATUS"), "w", encoding="utf-8") as file_h:
                file_h.write("a:b\na: 09:00:00 .... 09:00:01")

    with open("runpathfile", "w", encoding="utf-8") as file_h:
        for iteration in iters:
            for real in reals:
                runpath = os.path.join(f"realization-{real}", f"iter-{iteration}")
                file_h.write(f"{real:03d} {runpath} RES_{real} {iteration:03d}\n")


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_that_a_not_found_realization_is_skipped(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    shutil.rmtree("realization-1/iter-1")
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + KEYWORDS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
        },
        summary,
    )


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_that_a_failed_realization_is_skipped(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    os.remove("realization-0/iter-1/RES_0.SMSPEC")
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + KEYWORDS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 1),
        },
        summary,
    )


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_that_a_missing_realization_index_is_ok(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    rp_lines = Path("runpathfile").read_text(encoding="utf-8").splitlines()
    Path("sliced_runpathfile").write_text(
        rp_lines[1] + "\n" + rp_lines[3], encoding="utf-8"
    )
    csv_export2.csv_exporter(
        runpathfile="sliced_runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + KEYWORDS + ["FOO"],
        {
            ("iter-0", 1),
            ("iter-1", 1),
        },
        summary,
    )


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_that_iterations_in_runpathfile_cannot_be_defaulted(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    Path("runpathfile").write_text(
        "000 realization-0/iter-0 RES_0\n001 realization-0/iter-1 RES_1\n",
        encoding="utf-8",
    )

    with pytest.raises(UserWarning):
        csv_export2.csv_exporter(
            runpathfile="runpathfile",
            time_index="yearly",
            outputfile="unsmry--yearly.csv",
            column_keys=["F?PT"],
        )


def test_empty_file_yields_user_warning():
    with open("empty_file", "a", encoding="utf-8") as empty_file, pytest.raises(
        UserWarning, match="No data found"
    ):
        csv_export2.csv_exporter(
            runpathfile=empty_file.name,
            time_index="raw",
            outputfile="unsmry--yearly.csv",
            column_keys=["*"],
        )


@pytest.mark.parametrize("input_rst", [csv_export2.DESCRIPTION, csv_export2.EXAMPLES])
def test_valid_rst(input_rst):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck_core.checker.check_source(input_rst))


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_export_ensemble(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + KEYWORDS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
        summary,
    )


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(yearly_summary)
def test_export_ensemble_noparams(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=False)
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["FOPT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE", "FOPT"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
        summary,
    )


def verify_exported_file(
    exported_file_name, result_header, result_iter_rel, summary=None
):
    """Verify an exported CSV file with respect to:

        * Exactly the set of requested headers is found
        * The realizations and iterations that exist must equal
          given set of tuples.

    Args:
        exported_file_name (str): path to CSV file.
        result_header (list of str): The strings required in the header.
        result_iter_real (set): Set of 2-tuples: {(iterstring, realidx)}
    """
    dframe = pd.read_csv(exported_file_name)
    assert set(dframe.columns) == set(result_header)
    assert (
        set(dframe[["ENSEMBLE", "REAL"]].itertuples(index=False, name=None))
        == result_iter_rel
    )
    if summary:
        smspec, unsmry = summary
        keys = (set(dframe.columns) - {"ENSEMBLE", "REAL", "DATE"}).intersection(
            KEYWORDS
        )
        ensemble, real = list(result_iter_rel)[0]
        for k in keys:
            dframe.sort_values(by=["DATE"])
            dframe = dframe[(dframe["ENSEMBLE"] == ensemble) & (dframe["REAL"] == real)]
            assert dframe[k].tolist() == pytest.approx(
                [
                    m.params[smspec.keywords.index(k)]
                    for s in unsmry.steps
                    for m in s.ministeps
                ]
            )


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(monthly_summary)
def test_ert_integration(summary):
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    with open("FOO.DATA", "w", encoding="utf-8") as file_h:
        file_h.write("--Empty")

    with open("wf_csvexport", "w", encoding="utf-8") as file_h:
        file_h.write(
            # This workflow is representing the example in csv_export2.py:
            "MAKE_DIRECTORY csv_output\n"
            "EXPORT_RUNPATH * | *\n"  # (not really relevant in mocked case)
            "CSV_EXPORT2 runpathfile csv_output/data.csv monthly FOPT\n"
            # Example in documentation uses <RUNPATH_FILE> which is
            # linked to the RUNPATH keyword that we don't use in this
            # test (mocking data gets more complex if that is to be used)
        )

    ert_config = Path("test.ert")
    ert_config.write_text(
        dedent(
            """
            ECLBASE FOO.DATA
            QUEUE_SYSTEM LOCAL
            NUM_REALIZATIONS 2
            LOAD_WORKFLOW wf_csvexport
            HOOK_WORKFLOW wf_csvexport PRE_SIMULATION
            """
        )
    )
    with ErtPluginContext() as ctx:
        run_cli_with_pm(
            ["test_run", "--disable-monitor", str(ert_config)],
            ctx.plugin_manager,
        )

    assert pd.read_csv("csv_output/data.csv").shape == (16, 5)


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
@given(monthly_summary)
def test_ert_integration_errors(summary):
    """Test CSV_EXPORT2 when runpathfile points to non-existing realizations

    Tests that CSV_EXPORT2 happily skips non-existing
    realizations, but emits a warning that there is no STATUS file.
    """
    mock_data(summary, reals=[0, 1], iters=[0, 1], parameters=True)
    with open("FOO.DATA", "w", encoding="utf-8") as file_h:
        file_h.write("--Empty")

    # Append a not-existing realizations to the runpathfile:
    with open("runpathfile", "a", encoding="utf-8") as file_h:
        file_h.write("002 realization-2/iter-0 RES_1 000")

    with open("wf_csvexport", "w", encoding="utf-8") as file_h:
        file_h.write("CSV_EXPORT2 runpathfile data.csv monthly FOPT\n")

    ert_config = Path("test.ert")
    ert_config.write_text(
        dedent(
            """
            ECLBASE FOO.DATA
            QUEUE_SYSTEM LOCAL
            NUM_REALIZATIONS 2
            LOAD_WORKFLOW wf_csvexport
            HOOK_WORKFLOW wf_csvexport PRE_SIMULATION
            """
        )
    )
    with ErtPluginContext() as ctx:
        run_cli_with_pm(
            ["test_run", "--disable-monitor", str(ert_config)],
            ctx.plugin_manager,
        )

    assert os.path.exists("data.csv")
    verify_exported_file(
        "data.csv",
        ["ENSEMBLE", "REAL", "DATE", "FOPT", "FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
        summary,
    )


def test_csv_export2_job_is_loaded():
    pm = ErtPluginManager()
    assert "CSV_EXPORT2" in pm.get_installable_workflow_jobs()
