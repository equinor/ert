import json
import os
import random
import stat
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from ert.cli.main import ErtCliError
from ert.config import ErtConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
)
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
    config_path = ErtConfig.from_file("poly.ert").config_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        params = experiment.get_ensemble_by_name("default").load_parameters(
            "DESIGN_MATRIX"
        )["values"]
        np.testing.assert_array_equal(params[:, 0], [str(idx) for idx in a_values])
        np.testing.assert_array_equal(params[:, 1], 5 * ["cat1"] + 5 * ["cat2"])
        np.testing.assert_array_equal(params[:, 2], 10 * ["1"])
        np.testing.assert_array_equal(params[:, 3], 10 * ["2"])

    real_0_iter_0_parameters_json_path = (
        Path(config_path) / "poly_out" / "realization-0" / "iter-0" / "parameters.json"
    )
    assert real_0_iter_0_parameters_json_path.exists()
    with open(real_0_iter_0_parameters_json_path, mode="r+", encoding="utf-8") as fs:
        parameters_contents = json.load(fs)
    assert isinstance(parameters_contents, dict)
    design_matrix_content = parameters_contents.get("DESIGN_MATRIX")
    assert isinstance(design_matrix_content, dict)
    for k, v in design_matrix_content.items():
        if k == "category":
            assert isinstance(v, str)
        else:
            assert isinstance(v, float | int)


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
                GEN_KW COEFFS my_template my_output coeff_priors
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

    with open("my_template", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                a: <a>
                b: <b>
                c: <c>
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
    with open("poly_out/realization-0/iter-0/my_output", encoding="utf-8") as f:
        output = [line.strip() for line in f.readlines()]
    assert output[0] == "a: 0"
    assert output[1] == "b: 1"
    assert output[2] == "c: 2"


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


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    "experiment_mode, ensemble_name, iterations",
    [
        (ES_MDA_MODE, "default_", 4),
        (ENSEMBLE_SMOOTHER_MODE, "iter-", 2),
    ],
)
def test_design_matrix_on_esmda(experiment_mode, ensemble_name, iterations):
    design_path = "design_matrix.xlsx"
    reals = range(10)
    values = [random.uniform(0, 2) for _ in reals]
    _create_design_matrix(
        design_path,
        pd.DataFrame(
            {
                "REAL": list(range(10)),
                "b": values,
            }
        ),
        pd.DataFrame([]),
    )

    with open("poly.ert", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                QUEUE_OPTION LOCAL MAX_RUNNING 50
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                OBS_CONFIG observations
                NUM_REALIZATIONS 10
                GEN_KW COEFFS_A coeff_priors_a
                GEN_KW COEFFS_B coeff_priors_b
                GEN_KW COEFFS_C coeff_priors_c
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                DESIGN_MATRIX design_matrix.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
            )
        )

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python3
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        params = json.load(f)
                        params = params["COEFFS_A"] | params["COEFFS_B"] | params["COEFFS_C"]
                        return params

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

    with open("coeff_priors_a", "w", encoding="utf-8") as f:
        f.write("a UNIFORM 0 1")
    with open("coeff_priors_b", "w", encoding="utf-8") as f:
        f.write("b UNIFORM 0 2")
    with open("coeff_priors_c", "w", encoding="utf-8") as f:
        f.write("c UNIFORM 0 5")

    run_cli(
        experiment_mode,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
    )
    storage_path = ErtConfig.from_file("poly.ert").ens_path
    coeffs_a_previous = None
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        for i in range(iterations):
            ensemble = experiment.get_ensemble_by_name(f"{ensemble_name}{i}")

            # coeffs_a should be different in all realizations
            coeffs_a = ensemble.load_parameters("COEFFS_A")["values"].values.flatten()
            if coeffs_a_previous is not None:
                assert not np.array_equal(coeffs_a, coeffs_a_previous)
            coeffs_a_previous = coeffs_a

            # ceffs_b should be overridden by design matrix and be the same for all realizations
            coeffs_b = ensemble.load_parameters("COEFFS_B")["values"].values.flatten()
            assert values == pytest.approx(coeffs_b, 0.0001)
