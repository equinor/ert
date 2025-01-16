import os
import stat
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from ert.cli.main import ErtCliError
from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


def _create_design_matrix(filename, design_sheet_df, default_sheet_df=None):
    with pd.ExcelWriter(filename) as xl_write:
        design_sheet_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        if default_sheet_df is not None:
            default_sheet_df.to_excel(
                xl_write, index=False, sheet_name="DefaultSheet", header=False
            )


@pytest.mark.usefixtures("copy_poly_case")
def test_run_poly_example_with_design_matrix():
    num_realizations = 10
    a_values = list(range(num_realizations))
    _create_design_matrix(
        "poly_design.xlsx",
        pd.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": a_values,
                "category": 5 * ["cat1"] + 5 * ["cat2"],
            }
        ),
        pd.DataFrame([["b", 1], ["c", 2]]),
    )

    with open("poly.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """\
                QUEUE_OPTION LOCAL MAX_RUNNING 10
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                NUM_REALIZATIONS 10
                MIN_REALIZATIONS 1
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
            )
        )

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["DESIGN_MATRIX"]

                def _evaluate(coeffs, x):
                    assert coeffs["category"] in ["cat1", "cat2"]
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    coeffs = _load_coeffs("parameters.json")
                    output = [_evaluate(coeffs, x) for x in range(10)]
                    with open("poly.out", "w", encoding="utf-8") as f:
                        f.write("\\n".join(map(str, output)))
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
    )
    storage_path = ErtConfig.from_file("poly.ert").ens_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        params = experiment.get_ensemble_by_name("default").load_parameters(
            "DESIGN_MATRIX"
        )["values"]
        # All parameters are strings, this needs to be fixed
        np.testing.assert_array_equal(params[:, 0], [str(idx) for idx in a_values])
        np.testing.assert_array_equal(params[:, 1], 5 * ["cat1"] + 5 * ["cat2"])
        np.testing.assert_array_equal(params[:, 2], 10 * ["1"])
        np.testing.assert_array_equal(params[:, 3], 10 * ["2"])


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    "default_values, error_msg",
    [
        ([["b", 1], ["c", 2]], None),
        ([["b", 1]], "Overlapping parameter names found in design matrix!"),
    ],
)
def test_run_poly_example_with_design_matrix_and_genkw_merge(default_values, error_msg):
    num_realizations = 10
    a_values = list(range(num_realizations))
    _create_design_matrix(
        "poly_design.xlsx",
        pd.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": a_values,
            }
        ),
        pd.DataFrame(default_values),
    )

    with open("poly.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """\
                QUEUE_OPTION LOCAL MAX_RUNNING 10
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                NUM_REALIZATIONS 10
                MIN_REALIZATIONS 1
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                GEN_KW COEFFS coeff_priors
                DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
            )
        )

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["COEFFS"]

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    coeffs = _load_coeffs("parameters.json")
                    output = [_evaluate(coeffs, x) for x in range(10)]
                    with open("poly.out", "w", encoding="utf-8") as f:
                        f.write("\\n".join(map(str, output)))
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    if error_msg:
        with pytest.raises(ErtCliError, match=error_msg):
            run_cli(
                ENSEMBLE_EXPERIMENT_MODE,
                "--disable-monitoring",
                "poly.ert",
                "--experiment-name",
                "test-experiment",
            )
        return
    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
    )
    storage_path = ErtConfig.from_file("poly.ert").ens_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        params = experiment.get_ensemble_by_name("default").load_parameters("COEFFS")[
            "values"
        ]
        np.testing.assert_array_equal(params[:, 0], a_values)
        np.testing.assert_array_equal(params[:, 1], 10 * [1])
        np.testing.assert_array_equal(params[:, 2], 10 * [2])


@pytest.mark.usefixtures("copy_poly_case")
def test_run_poly_example_with_multiple_design_matrix_instances():
    num_realizations = 10
    a_values = list(range(num_realizations))
    _create_design_matrix(
        "poly_design_1.xlsx",
        pd.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": a_values,
            }
        ),
        pd.DataFrame([["b", 1], ["c", 2]]),
    )
    _create_design_matrix(
        "poly_design_2.xlsx",
        pd.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "d": num_realizations * [3],
            }
        ),
        pd.DataFrame([["g", 4]]),
    )

    with open("poly.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """\
                QUEUE_OPTION LOCAL MAX_RUNNING 10
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                NUM_REALIZATIONS 10
                MIN_REALIZATIONS 1
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                DESIGN_MATRIX poly_design_1.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultSheet
                DESIGN_MATRIX poly_design_2.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
            )
        )

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["DESIGN_MATRIX"]

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    coeffs = _load_coeffs("parameters.json")
                    output = [_evaluate(coeffs, x) for x in range(10)]
                    with open("poly.out", "w", encoding="utf-8") as f:
                        f.write("\\n".join(map(str, output)))
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
    )
    storage_path = ErtConfig.from_file("poly.ert").ens_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        params = experiment.get_ensemble_by_name("default").load_parameters(
            "DESIGN_MATRIX"
        )["values"]
        np.testing.assert_array_equal(params[:, 0], a_values)
        np.testing.assert_array_equal(params[:, 1], 10 * [1])
        np.testing.assert_array_equal(params[:, 2], 10 * [2])
        np.testing.assert_array_equal(params[:, 3], 10 * [3])
        np.testing.assert_array_equal(params[:, 4], 10 * [4])
