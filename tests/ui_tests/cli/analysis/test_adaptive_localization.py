import shutil
from textwrap import dedent

import numpy as np
import pytest

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

random_seed_line = "RANDOM_SEED 1234\n\n"
from tests.ui_tests.cli.run_cli import run_cli


def run_cli_ES_with_case(poly_config):
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "--realizations",
        "1-50",
        poly_config,
        "--port-range",
        "1024-65535",
        "--experiment-name",
        "test-experiment",
    )
    storage_path = ErtConfig.from_file(poly_config).ens_path
    with open_storage(storage_path) as storage:
        experiment = storage.get_experiment_by_name("test-experiment")
        prior_ensemble = experiment.get_ensemble_by_name("iter-0")
        posterior_ensemble = experiment.get_ensemble_by_name("iter-1")
    return prior_ensemble, posterior_ensemble


@pytest.mark.usefixtures("copy_poly_case")
def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior():
    set_adaptive_localization_1 = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 1.0
        """
    )

    with open("poly.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        lines.insert(2, random_seed_line)
        lines.insert(9, set_adaptive_localization_1)

    with open("poly_localization_1.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)
    prior_ensemble, posterior_ensemble = run_cli_ES_with_case("poly_localization_1.ert")

    with pytest.raises(
        FileNotFoundError, match="No cross-correlation data available at"
    ):
        prior_ensemble.load_cross_correlations()

    prior_sample = prior_ensemble.load_parameters("COEFFS")["values"]
    posterior_sample = posterior_ensemble.load_parameters("COEFFS")["values"]
    # Check prior and posterior samples are equal
    assert np.allclose(posterior_sample, prior_sample)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_adaptive_localization_works_with_a_single_observation():
    """This is a regression test as ert would crash if adaptive localization
    was run with a single observation.
    """
    set_adaptive_localization_0 = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0
        """
    )

    with open("poly.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        lines.insert(9, set_adaptive_localization_0)

    content = """GENERAL_OBSERVATION POLY_OBS {
        DATA       = POLY_RES;
        INDEX_LIST = 0;
        OBS_FILE   = poly_obs_data.txt;
    };"""

    with open("observations", "w", encoding="utf-8") as file:
        file.write(content)

    content = "2.1457049781272213 0.6"

    with open("poly_obs_data.txt", "w", encoding="utf-8") as file:
        file.write(content)

    with open("poly_localization_0.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    _, _ = run_cli_ES_with_case("poly_localization_0.ert")


@pytest.mark.usefixtures("copy_poly_case")
def test_that_adaptive_localization_works_with_multiple_observations(snapshot):
    with open("observations", "w", encoding="utf-8") as file:
        file.write(
            """GENERAL_OBSERVATION POLY_OBS {
        DATA       = POLY_RES;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_1 {
        DATA       = POLY_RES1;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data1.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_2 {
        DATA       = POLY_RES2;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data2.txt;
    };
    """
        )

    with open("poly_eval.py", "w", encoding="utf-8") as file:
        file.write(
            """#!/usr/bin/env python3
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

    with open("poly.out1", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*2 for x in output])))

    with open("poly.out2", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*3 for x in output])))
"""
        )

    shutil.copy("poly_obs_data.txt", "poly_obs_data1.txt")
    shutil.copy("poly_obs_data.txt", "poly_obs_data2.txt")

    with open("poly_localization_0.ert", "w", encoding="utf-8") as f:
        f.write(
            """
        QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 12

RUNPATH poly_out/realization-<IENS>/iter-<ITER>

OBS_CONFIG observations
REALIZATION_MEMORY 50mb

NUM_REALIZATIONS 100
MIN_REALIZATIONS 1

GEN_KW COEFFS coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly.out
GEN_DATA POLY_RES1 RESULT_FILE:poly.out1
GEN_DATA POLY_RES2 RESULT_FILE:poly.out2

INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval

ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0

ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *
ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE POLY_OBS1_*
"""
        )

    expected_records = {
        ("*", "POLY_OBS", "0, 0"),
        ("*", "POLY_OBS", "0, 1"),
        ("*", "POLY_OBS", "0, 2"),
        ("*", "POLY_OBS", "0, 3"),
        ("*", "POLY_OBS", "0, 4"),
        ("*", "POLY_OBS1_1", "0, 0"),
        ("*", "POLY_OBS1_1", "0, 1"),
        ("*", "POLY_OBS1_1", "0, 2"),
        ("*", "POLY_OBS1_1", "0, 3"),
        ("*", "POLY_OBS1_1", "0, 4"),
        ("*", "POLY_OBS1_2", "0, 0"),
        ("*", "POLY_OBS1_2", "0, 1"),
        ("*", "POLY_OBS1_2", "0, 2"),
        ("*", "POLY_OBS1_2", "0, 3"),
        ("*", "POLY_OBS1_2", "0, 4"),
        ("POLY_OBS1_*", "POLY_OBS1_1", "0, 0"),
        ("POLY_OBS1_*", "POLY_OBS1_1", "0, 1"),
        ("POLY_OBS1_*", "POLY_OBS1_1", "0, 2"),
        ("POLY_OBS1_*", "POLY_OBS1_1", "0, 3"),
        ("POLY_OBS1_*", "POLY_OBS1_1", "0, 4"),
        ("POLY_OBS1_*", "POLY_OBS1_2", "0, 0"),
        ("POLY_OBS1_*", "POLY_OBS1_2", "0, 1"),
        ("POLY_OBS1_*", "POLY_OBS1_2", "0, 2"),
        ("POLY_OBS1_*", "POLY_OBS1_2", "0, 3"),
        ("POLY_OBS1_*", "POLY_OBS1_2", "0, 4"),
    }

    prior_ens, _ = run_cli_ES_with_case("poly_localization_0.ert")
    sf = prior_ens.load_observation_scaling_factors()
    set_of_records_from_xr = {
        x[:-1]
        for x in sf.to_dataframe()
        .reset_index()
        .set_index(["input_group", "obs_key", "index"])
        .dropna()
        .to_records()
        .tolist()
    }

    assert set_of_records_from_xr == expected_records


@pytest.mark.usefixtures("copy_poly_case")
def test_that_adaptive_localization_with_cutoff_0_equals_ESupdate():
    """
    Note that "RANDOM_SEED" in both ert configs needs to be the same to obtain
    the same sample from the prior.
    """
    set_adaptive_localization_0 = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0
        """
    )

    with open("poly.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        lines.insert(2, random_seed_line)

    with open("poly_no_localization.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    lines.insert(9, set_adaptive_localization_0)

    with open("poly_localization_0.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    _, posterior_ensemble_loc0 = run_cli_ES_with_case("poly_localization_0.ert")
    _, posterior_ensemble_noloc = run_cli_ES_with_case("poly_no_localization.ert")

    posterior_sample_loc0 = posterior_ensemble_loc0.load_parameters("COEFFS")["values"]
    posterior_sample_noloc = posterior_ensemble_noloc.load_parameters("COEFFS")[
        "values"
    ]

    # Check posterior sample without adaptive localization and with cut-off 0 are equal
    assert np.allclose(posterior_sample_loc0, posterior_sample_noloc)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_posterior_generalized_variance_increases_in_cutoff():
    rng = np.random.default_rng(42)
    cutoff1 = rng.uniform(0, 1)
    cutoff2 = rng.uniform(cutoff1, 1)

    set_adaptive_localization_cutoff1 = dedent(
        f"""
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD {cutoff1}
        """
    )
    set_adaptive_localization_cutoff2 = dedent(
        f"""
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD {cutoff2}
        """
    )

    with open("poly.ert", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "NUM_REALIZATIONS 100" in line:
                lines[i] = "NUM_REALIZATIONS 200\n"
                break
        lines.insert(2, random_seed_line)
        lines.insert(9, set_adaptive_localization_cutoff1)

    with open("poly_localization_cutoff1.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    lines.remove(set_adaptive_localization_cutoff1)
    lines.insert(9, set_adaptive_localization_cutoff2)
    with open("poly_localization_cutoff2.ert", "w", encoding="utf-8") as f:
        f.writelines(lines)

    prior_ensemble_cutoff1, posterior_ensemble_cutoff1 = run_cli_ES_with_case(
        "poly_localization_cutoff1.ert"
    )
    _, posterior_ensemble_cutoff2 = run_cli_ES_with_case(
        "poly_localization_cutoff2.ert"
    )

    cross_correlations = prior_ensemble_cutoff1.load_cross_correlations()
    assert all(cross_correlations.parameter.to_numpy() == ["a", "b"])
    assert cross_correlations["COEFFS"].values.shape == (2, 5)
    assert (
        (cross_correlations["COEFFS"].values >= -1)
        & (cross_correlations["COEFFS"].values <= 1)
    ).all()

    prior_sample_cutoff1 = prior_ensemble_cutoff1.load_parameters("COEFFS")["values"]
    prior_cov = np.cov(prior_sample_cutoff1, rowvar=False)
    posterior_sample_cutoff1 = posterior_ensemble_cutoff1.load_parameters("COEFFS")[
        "values"
    ]
    posterior_cutoff1_cov = np.cov(posterior_sample_cutoff1, rowvar=False)
    posterior_sample_cutoff2 = posterior_ensemble_cutoff2.load_parameters("COEFFS")[
        "values"
    ]
    posterior_cutoff2_cov = np.cov(posterior_sample_cutoff2, rowvar=False)

    generalized_variance_1 = np.linalg.det(posterior_cutoff1_cov)
    generalized_variance_2 = np.linalg.det(posterior_cutoff2_cov)
    generalized_variance_prior = np.linalg.det(prior_cov)

    # Check that posterior generalized variance in positive, increases in cutoff and
    # does not exceed prior generalized variance
    assert generalized_variance_1 > 0, f"Assertion failed with cutoff1={cutoff1}"
    assert (
        generalized_variance_1 <= generalized_variance_2
    ), f"Assertion failed with cutoff1={cutoff1} and cutoff2={cutoff2}"
    assert (
        generalized_variance_2 <= generalized_variance_prior
    ), f"Assertion failed with cutoff2={cutoff2}"
