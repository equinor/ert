import os
import stat
from pathlib import Path
from textwrap import dedent

import numpy as np
import polars as pl
import pytest
from scipy.ndimage import gaussian_filter
from xtgeo import RegularSurface, surface_from_file

from ert.analysis._update_commons import _all_parameters
from ert.config import ErtConfig, GenKwConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import RealizationStorageState, open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
        ],
        update=True,
    )


@pytest.fixture
def obs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "response_key": "RESPONSE",
            "observation_key": "OBSERVATION",
            "report_step": pl.Series(np.full(3, 0), dtype=pl.UInt16),
            "index": pl.Series([0, 1, 2], dtype=pl.UInt16),
            "observations": pl.Series([1.0, 1.0, 1.0], dtype=pl.Float32),
            "std": pl.Series([0.1, 1.0, 10.0], dtype=pl.Float32),
        }
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_that_posterior_has_lower_variance_than_prior():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "--realizations",
        "1-50",
        "poly.ert",
        "--experiment-name",
        "es-test",
    )
    with open_storage("storage") as storage:
        experiment = storage.get_experiment_by_name("es-test")
        prior_ensemble = experiment.get_ensemble_by_name("iter-0")
        df_default = prior_ensemble.load_all_gen_kw_data()
        posterior_ensemble = experiment.get_ensemble_by_name("iter-1")
        df_target = posterior_ensemble.load_all_gen_kw_data()

        # The std for the ensemble should decrease
        assert float(
            prior_ensemble.calculate_std_dev_for_parameter_group("COEFFS").sum()
        ) > float(
            posterior_ensemble.calculate_std_dev_for_parameter_group("COEFFS").sum()
        )

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_target.cov().to_numpy())
        < np.linalg.det(df_default.cov().to_numpy())
    )


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_that_surfaces_retain_their_order_when_loaded_and_saved_by_ert():
    """This is a regression test to make sure ert does not use the wrong order
    (row-major / column-major) when working with surfaces.
    """
    rng = np.random.default_rng()

    def sample_prior(nx, ny):
        return np.exp(
            5
            * gaussian_filter(
                gaussian_filter(rng.random(size=(nx, ny)), sigma=2.0), sigma=1.0
            )
        )

    nx = 5
    ny = 7
    ensemble_size = 2

    Path("./surface").mkdir()
    for i in range(ensemble_size):
        surf = RegularSurface(
            ncol=nx, nrow=ny, xinc=1.0, yinc=1.0, values=sample_prior(nx, ny)
        )
        surf.to_file(f"surface/surf_init_{i}.irap", fformat="irap_ascii")

    # Single observation with a large ERROR to make sure the udpate is minimal.
    obs = """
    SUMMARY_OBSERVATION WOPR_OP1_9
    {
        VALUE   = 0.1;
        ERROR   = 200.0;
        DATE    = 2010-03-31;
        KEY     = WOPR:OP1;
    };
    """

    with open("observations/observations.txt", "w", encoding="utf-8") as file:
        file.write(obs)

    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "snake_oil_surface.ert",
    )

    ert_config = ErtConfig.from_file("snake_oil_surface.ert")

    storage = open_storage(ert_config.ens_path)
    experiment = storage.get_experiment_by_name("es")
    ens_prior = experiment.get_ensemble_by_name("iter-0")
    ens_posterior = experiment.get_ensemble_by_name("iter-1")

    # Check that surfaces defined in INIT_FILES are not changed by ERT
    surf_prior = ens_prior.load_parameters("TOP", list(range(ensemble_size)))["values"]
    for i in range(ensemble_size):
        prior_init = surface_from_file(
            f"surface/surf_init_{i}.irap", fformat="irap_ascii", dtype=np.float32
        )
        np.testing.assert_array_equal(surf_prior[i], prior_init.values.data)

    surf_posterior = ens_posterior.load_parameters("TOP", list(range(ensemble_size)))[
        "values"
    ]

    assert surf_prior.shape == surf_posterior.shape

    for i in range(ensemble_size):
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(surf_prior[i], surf_posterior[i])
        np.testing.assert_almost_equal(
            surf_prior[i].values, surf_posterior[i].values, decimal=2
        )


@pytest.mark.usefixtures("copy_snake_oil_field")
def test_update_multiple_param():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "snake_oil.ert",
    )

    ert_config = ErtConfig.from_file("snake_oil.ert")

    storage = open_storage(ert_config.ens_path)
    experiment = storage.get_experiment_by_name("es")
    prior_ensemble = experiment.get_ensemble_by_name("iter-0")
    posterior_ensemble = experiment.get_ensemble_by_name("iter-1")

    prior_array = _all_parameters(prior_ensemble, list(range(10)))
    posterior_array = _all_parameters(posterior_ensemble, list(range(10)))

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    # https://en.wikipedia.org/wiki/Variance#For_vector-valued_random_variables
    assert np.trace(np.cov(posterior_array)) < np.trace(np.cov(prior_array))


@pytest.mark.usefixtures("copy_poly_case")
def test_that_reals_with_load_failure_in_prior_become_parent_failure_in_posterior():
    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import numpy as np
                import sys
                import json
                import os

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["COEFFS"]

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    # Kill one realization per iteration
                    if (
                        os.getenv("_ERT_REALIZATION_NUMBER") ==
                        os.getenv("_ERT_ITERATION_NUMBER")
                        ):
                        sys.exit(1)
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
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "--realizations",
        "0-2",
        "poly.ert",
    )

    ert_config = ErtConfig.from_file("poly.ert")

    with open_storage(ert_config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("es")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")

        assert all(
            RealizationStorageState.FAILURE_IN_PARENT
            in posterior.get_ensemble_state()[idx]
            for idx, v in enumerate(prior.get_ensemble_state())
            if RealizationStorageState.FAILURE_IN_CURRENT in v
        )
