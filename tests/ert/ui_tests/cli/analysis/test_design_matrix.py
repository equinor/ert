import json
import logging
import os
import random
import stat
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import polars as pl
import pytest

from ert.cli.main import ErtCliError
from ert.config import ConfigWarning, ErtConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
)
from ert.storage import open_storage
from tests.ert.conftest import _create_design_matrix
from tests.ert.ui_tests.cli.run_cli import run_cli


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_run_poly_example_with_design_matrix(copy_poly_case_with_design_matrix, caplog):
    caplog.set_level(logging.INFO)
    num_realizations = 10
    a_values = list(range(num_realizations))
    design_dict = {
        "REAL": list(range(num_realizations)),
        "a": a_values,
        "category": 5 * ["cat1"] + 5 * ["cat2"],
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)

    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
    )

    for param in ["a", "b", "c", "category"]:
        assert f"Getting parameter {param} from design matrix" in caplog.text

    storage_path = ErtConfig.from_file("poly.ert").ens_path
    config_path = ErtConfig.from_file("poly.ert").config_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        params = experiment.get_ensemble_by_name("default").load_parameters(
            "DESIGN_MATRIX"
        )
        np.testing.assert_array_equal(params["a"].to_list(), a_values)
        np.testing.assert_array_equal(
            params["category"].to_list(), 5 * ["cat1"] + 5 * ["cat2"]
        )
        np.testing.assert_array_equal(params["b"].to_list(), 10 * [1])
        np.testing.assert_array_equal(params["c"].to_list(), 10 * [2])

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


@pytest.mark.usefixtures(
    "copy_poly_case", "use_site_configurations_with_no_queue_options"
)
@pytest.mark.parametrize(
    "default_values",
    [
        ([["b", 1], ["c", 2]]),
    ],
)
def test_run_poly_example_with_design_matrix_and_genkw_merge(default_values):
    num_realizations = 10
    a_values = list(range(num_realizations))
    _create_design_matrix(
        "poly_design.xlsx",
        pl.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": a_values,
                "category": 5 * ["cat1"] + 5 * ["cat2"],
            }
        ),
        pl.DataFrame(default_values, orient="row"),
    )

    Path("poly.ert").write_text(
        dedent(
            """\
                QUEUE_OPTION LOCAL MAX_RUNNING 2
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                NUM_REALIZATIONS 10
                MIN_REALIZATIONS 1
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                GEN_KW COEFFS my_template my_output coeff_priors
                DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet \
                    DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
        ),
        encoding="utf-8",
    )

    # This adds a dummy category parameter to COEFFS GENKW
    # which will be overridden by the design matrix catagorical entries
    with open("coeff_priors", "a", encoding="utf-8") as f:
        f.write("category UNIFORM 0 1")

    Path("poly_eval.py").write_text(
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
        ),
        encoding="utf-8",
    )

    Path("my_template").write_text(
        dedent(
            """\
                a: <a>
                b: <b>
                c: <c>
                category: <category>
                """
        ),
        encoding="utf-8",
    )

    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConfigWarning)
        # Expected warning:
        # ConfigWarning: Parameters {'c', 'a', 'b', 'category'} from GEN_KW
        # group 'COEFFS' will be overridden by design matrix.
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
        params = experiment.get_ensemble_by_name("default").load_parameters("COEFFS")
        np.testing.assert_array_equal(params["a"].to_list(), a_values)
        np.testing.assert_array_equal(
            params["category"].to_list(), 5 * ["cat1"] + 5 * ["cat2"]
        )
        np.testing.assert_array_equal(params["b"].to_list(), 10 * [1])
        np.testing.assert_array_equal(params["c"].to_list(), 10 * [2])
    with open("poly_out/realization-0/iter-0/my_output", encoding="utf-8") as f:
        output = [line.strip() for line in f]
        assert output[0] == "a: 0"
        assert output[1] == "b: 1"
        assert output[2] == "c: 2"
        assert output[3] == "category: cat1"
    with open("poly_out/realization-5/iter-0/my_output", encoding="utf-8") as f:
        output = [line.strip() for line in f]
        assert output[0] == "a: 5"
        assert output[1] == "b: 1"
        assert output[2] == "c: 2"
        assert output[3] == "category: cat2"


@pytest.mark.usefixtures(
    "copy_poly_case", "use_site_configurations_with_no_queue_options"
)
def test_run_poly_example_with_multiple_design_matrix_instances():
    num_realizations = 10
    a_values = list(range(num_realizations))
    _create_design_matrix(
        "poly_design_1.xlsx",
        pl.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": a_values,
            }
        ),
        pl.DataFrame([["b", 1], ["c", 2]], orient="row"),
    )
    _create_design_matrix(
        "poly_design_2.xlsx",
        pl.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "d": num_realizations * [3],
            }
        ),
        pl.DataFrame([["g", 4]], orient="row"),
    )

    Path("poly.ert").write_text(
        dedent(
            """\
                QUEUE_OPTION LOCAL MAX_RUNNING 2
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                NUM_REALIZATIONS 10
                MIN_REALIZATIONS 1
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                DESIGN_MATRIX poly_design_1.xlsx DEFAULT_SHEET:DefaultSheet
                DESIGN_MATRIX poly_design_2.xlsx DEFAULT_SHEET:DefaultSheet
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
        ),
        encoding="utf-8",
    )

    Path("poly_eval.py").write_text(
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
        ),
        encoding="utf-8",
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
        )
        np.testing.assert_array_equal(params["a"].to_list(), a_values)
        np.testing.assert_array_equal(params["b"].to_list(), 10 * [1])
        np.testing.assert_array_equal(params["c"].to_list(), 10 * [2])
        np.testing.assert_array_equal(params["d"].to_list(), 10 * [3])
        np.testing.assert_array_equal(params["g"].to_list(), 10 * [4])


@pytest.mark.usefixtures(
    "copy_poly_case", "use_site_configurations_with_no_queue_options"
)
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
        pl.DataFrame(
            {
                "REAL": list(range(10)),
                "b": values,
            }
        ),
    )

    Path("poly.ert").write_text(
        dedent(
            """\
                QUEUE_OPTION LOCAL MAX_RUNNING 2
                RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                OBS_CONFIG observations
                NUM_REALIZATIONS 10
                GEN_KW COEFFS_A coeff_priors_a
                GEN_KW COEFFS_B coeff_priors_b
                GEN_KW COEFFS_C coeff_priors_c
                GEN_DATA POLY_RES RESULT_FILE:poly.out
                DESIGN_MATRIX design_matrix.xlsx
                INSTALL_JOB poly_eval POLY_EVAL
                FORWARD_MODEL poly_eval
                """
        ),
        encoding="utf-8",
    )

    Path("poly_eval.py").write_text(
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
                """  # noqa: E501
        ),
        encoding="utf-8",
    )

    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    Path("coeff_priors_a").write_text("a UNIFORM 0 1", encoding="utf-8")
    Path("coeff_priors_b").write_text("b UNIFORM 0 2", encoding="utf-8")
    Path("coeff_priors_c").write_text("c UNIFORM 0 5", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConfigWarning)
        # Expected warning:
        # ConfigWarning: Parameters {'b'} from GEN_KW group 'COEFFS_B'
        # will be overridden by design matrix.
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
            coeffs_a = ensemble.load_parameters("COEFFS_A")["a"].to_list()

            if coeffs_a_previous is not None:
                assert not np.array_equal(coeffs_a, coeffs_a_previous)
            coeffs_a_previous = coeffs_a

            # coeffs_b should be overridden by design matrix and be the
            # same for all realizations
            coeffs_b = ensemble.load_parameters("COEFFS_B")["b"].to_list()
            assert values == pytest.approx(coeffs_b, 0.0001)


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_run_poly_example_with_design_matrix_selective_realizations(
    copy_poly_case_with_design_matrix, capsys
):
    num_realizations = 5
    a_values = list(range(num_realizations))
    design_dict = {
        "REAL": [0, 3, 5, 6, 10],
        "a": a_values,
        "category": 2 * ["cat1"] + 3 * ["cat2"],
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)

    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "poly.ert",
        "--experiment-name",
        "test-experiment",
        "--realizations",
        "0,4",
    )
    config_path = ErtConfig.from_file("poly.ert").config_path

    realizations_run = os.listdir(Path(config_path) / "poly_out")
    assert len(realizations_run) == 1
    assert "realization-0" in realizations_run

    assert (
        "Using realizations intersected between realizations specified "
        "and DESIGN_MATRIX (1)" in capsys.readouterr().out
    )


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    "experiment_mode",
    [
        (ES_MDA_MODE),
        (ENSEMBLE_SMOOTHER_MODE),
    ],
)
def test_design_matrix_on_esmda_fail_without_updateable_parameters(
    copy_poly_case_with_design_matrix, experiment_mode
):
    reals = range(10)
    copy_poly_case_with_design_matrix(
        {"REAL": list(reals), "a": [random.uniform(0, 2) for _ in reals]},
        [["b", 1], ["c", 2]],
    )

    with open("poly.ert", "a", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                GEN_KW COEFFS coeff_priors
                OBS_CONFIG observations
                """
            )
        )

    with (
        pytest.raises(
            ErtCliError,
            match="No parameters to update as all parameters were set to update:false!",
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", category=ConfigWarning)
        # Expected warning:
        # ConfigWarning: Parameters {'c', 'a', 'b'} from GEN_KW
        # group 'COEFFS' will be overridden
        run_cli(
            experiment_mode,
            "--disable-monitoring",
            "poly.ert",
            "--experiment-name",
            "test-experiment",
        )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "realizations_in_design_matrix, expected_message_format",
    [
        pytest.param(
            5,
            (
                r"NUM_REALIZATIONS \({num_realizations_in_user_config}\) is "
                "greater than the number of realizations in DESIGN_MATRIX "
                r"\({realizations_in_design_matrix}\)\. Using the realizations from "
                r"DESIGN_MATRIX \({realizations_in_design_matrix}\)"
            ),
            id="num_reals_greater_than_design_matrix",
        ),
        pytest.param(
            15,
            (
                r"NUM_REALIZATIONS \({num_realizations_in_user_config}\) is "
                "less than the number of realizations in DESIGN_MATRIX "
                r"\({realizations_in_design_matrix}\). Using the realizations from "
                r"NUM_REALIZATIONS \({num_realizations_in_user_config}\)"
            ),
            id="num_reals_less_than_design_matrix",
        ),
    ],
)
def test_run_poly_example_with_different_realization_count_chooses_smaller_and_warns(
    realizations_in_design_matrix,
    expected_message_format,
    copy_poly_case_with_design_matrix,
):
    num_realizations_in_user_config = 10
    expected_message = expected_message_format.format(
        num_realizations_in_user_config=num_realizations_in_user_config,
        realizations_in_design_matrix=realizations_in_design_matrix,
    )

    realization_list = list(range(realizations_in_design_matrix))
    design_dict = {
        "REAL": realization_list,
        "a": [2 * a for a in realization_list],
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)
    with open("poly.ert", "a+", encoding="utf-8") as f:
        f.write(f"\nNUM_REALIZATIONS {num_realizations_in_user_config}")
    with pytest.warns(ConfigWarning, match=expected_message):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "--disable-monitoring",
            "poly.ert",
            "--experiment-name",
            "test-experiment",
        )
    config_path = ErtConfig.from_file("poly.ert").config_path

    realizations_run = os.listdir(Path(config_path) / "poly_out")
    assert len(realizations_run) == min(
        realizations_in_design_matrix, num_realizations_in_user_config
    )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize(
    "specified_realizations, intersected_realizations_count",
    [
        pytest.param(
            "--realizations=3-6",
            2,
            id="specified_realizations_finds_and_uses_intersect",
        ),
        pytest.param(
            "--realizations=0-4",
            0,
            id="specified_realizations_are_not_intersecting_and_raises",
        ),
    ],
)
def test_run_poly_example_with_specified_realizations_finds_intersection_and_warns(
    specified_realizations: str,
    intersected_realizations_count: int,
    copy_poly_case_with_design_matrix,
    capsys,
):
    realization_list = [5, 6, 7, 9, 10]
    design_dict = {
        "REAL": realization_list,
        "a": [2 * a for a in realization_list],
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)
    with open("poly.ert", "a+", encoding="utf-8") as f:
        f.write("\nNUM_REALIZATIONS 20")
    if intersected_realizations_count > 0:
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            specified_realizations,
            "--disable-monitoring",
            "poly.ert",
            "--experiment-name",
            "test-experiment",
        )

        assert (
            "Using realizations intersected between realizations specified and "
            f"DESIGN_MATRIX ({intersected_realizations_count})"
        ) in capsys.readouterr().out

        config_path = ErtConfig.from_file("poly.ert").config_path
        realizations_run = os.listdir(Path(config_path) / "poly_out")
        assert len(realizations_run) == intersected_realizations_count
    else:
        with pytest.raises(
            ErtCliError,
            match=(
                "The specified realizations do not intersect with the active "
                r"realizations in the design matrix."
            ),
        ):
            run_cli(
                ENSEMBLE_EXPERIMENT_MODE,
                specified_realizations,
                "--disable-monitoring",
                "poly.ert",
                "--experiment-name",
                "test-experiment",
            )
