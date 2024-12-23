import os
import stat
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import numpy.testing
import polars as pl
import pytest
import resfo
import xtgeo

from ert.analysis import (
    smoother_update,
)
from ert.config import ErtConfig
from ert.config.analysis_config import UpdateSettings
from ert.config.analysis_module import ESSettings
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli


def test_shared_heat_equation_storage(heat_equation_storage):
    """The fixture heat_equation_storage runs the heat equation test case.
    This test verifies that results are as expected.
    """
    config = heat_equation_storage
    with open_storage(config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("es-mda")
        ensembles = [experiment.get_ensemble_by_name(f"default_{i}") for i in range(4)]

        param_config = config.ensemble_config.parameter_configs["COND"]

        # Check that generalized variance decreases across consecutive ensembles
        covariances = []
        for ensemble in ensembles:
            results = ensemble.load_parameters("COND")["values"]
            reshaped_values = results.values.reshape(
                ensemble.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            )
            covariances.append(np.cov(reshaped_values, rowvar=False))
        for i in range(len(covariances) - 1):
            assert np.trace(covariances[i]) > np.trace(
                covariances[i + 1]
            ), f"Generalized variance did not decrease from iteration {i} to {i + 1}"

        # Check that the saved cross-correlations are as expected.
        for i in range(3):
            ensemble = ensembles[i]
            corr_XY = ensemble.load_cross_correlations()

            assert sorted(corr_XY["param_group"].unique().to_list()) == [
                "CORR_LENGTH",
                "INIT_TEMP_SCALE",
            ]
            assert corr_XY["param_name"].unique().to_list() == ["x"]

            # Make sure correlations are between -1 and 1.
            is_valid = (corr_XY["value"] >= -1) & (corr_XY["value"] <= 1)
            assert is_valid.all()

            # Check obs names and obs groups
            expected_obs_groups = [obs[0] for obs in config.observation_config]
            obs_groups = corr_XY["obs_group"].unique().to_list()
            assert sorted(obs_groups) == sorted(expected_obs_groups)
            # Check that obs names are created using obs groups
            obs_name_starts_with_group = (
                corr_XY.with_columns(
                    pl.col("obs_name")
                    .str.starts_with(pl.col("obs_group"))
                    .alias("starts_with_check")
                )
                .get_column("starts_with_check")
                .all()
            )
            assert obs_name_starts_with_group

    # Check that fields in the runpath are different between ensembles
    cond_iter0 = resfo.read("simulations/realization-0/iter-0/cond.bgrdecl")[0][1]
    cond_iter1 = resfo.read("simulations/realization-0/iter-1/cond.bgrdecl")[0][1]
    assert (cond_iter0 != cond_iter1).all()


def test_parameter_update_with_inactive_cells_xtgeo_grdecl(tmpdir):
    """
    This replicates the poly example, only it uses FIELD parameter
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
            NUM_REALIZATIONS 5
            OBS_CONFIG observations
            FIELD MY_PARAM PARAMETER my_param.grdecl INIT_FILES:my_param.grdecl FORWARD_INIT:True
            GRID MY_EGRID.EGRID
            GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
            INSTALL_JOB poly_eval POLY_EVAL
            FORWARD_MODEL poly_eval
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        realizations = 5
        NCOL = 123
        NROW = 111
        NLAY = 6
        grid = xtgeo.create_box_grid(dimension=(NCOL, NROW, NLAY))
        mask = grid.get_actnum()
        rng = np.random.default_rng()
        mask_list = rng.choice([True, False], NCOL * NROW * NLAY)

        # make sure we filter out the 'c' parameter
        for i in range(NLAY):
            idx = i * NCOL * NROW
            mask_list[idx : idx + 3] = [True, True, False]

        mask.values = mask_list
        grid.set_actnum(mask)
        grid.to_file("MY_EGRID.EGRID", "egrid")

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""#!/usr/bin/env python
import xtgeo
import numpy as np
import os
if __name__ == "__main__":
    if not os.path.exists("my_param.grdecl"):
        values = np.random.standard_normal({NCOL}*{NROW}*{NLAY})
        with open("my_param.grdecl", "w") as fout:
            fout.write("MY_PARAM\\n")
            fout.write(" ".join([str(val) for val in values]) + " /\\n")
    with open("my_param.grdecl", "r") as fin:
        for line_nr, line in enumerate(fin):
            if line_nr == 1:
                a, b, c, *_ = line.split()
    output = [float(a) * x**2 + float(b) * x + float(c) for x in range(10)]
    with open("gen_data_0.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))
        """
                )
            )
        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        with open("POLY_EVAL", "w", encoding="utf-8") as fout:
            fout.write("EXECUTABLE forward_model")
        with open("observations", "w", encoding="utf-8") as fout:
            fout.write(
                dedent(
                    """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = MY_RESPONSE;
                INDEX_LIST = 0,2,4,6,8;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
                )
            )

        with open("obs.txt", "w", encoding="utf-8") as fobs:
            fobs.write(
                dedent(
                    """
            2.1457049781272213 0.6
            8.769219841380755 1.4
            12.388014786122742 3.0
            25.600464531354252 5.4
            42.35204755970952 8.6"""
                )
            )

        run_cli(
            ENSEMBLE_SMOOTHER_MODE,
            "--disable-monitor",
            "config.ert",
        )
        config = ErtConfig.from_file("config.ert")
        with open_storage(config.ens_path) as storage:
            experiment = storage.get_experiment_by_name("es")
            prior = experiment.get_ensemble_by_name("iter-0")
            posterior = experiment.get_ensemble_by_name("iter-1")

            prior_result = prior.load_parameters("MY_PARAM", list(range(realizations)))[
                "values"
            ]
            posterior_result = posterior.load_parameters(
                "MY_PARAM", list(range(realizations))
            )["values"]

            # check the shape of internal data used in the update
            assert prior_result.shape == (5, NCOL, NROW, NLAY)
            assert posterior_result.shape == (5, NCOL, NROW, NLAY)

            # Only assert on the first three rows, as there are only three parameters,
            # a, b and c, the rest have no correlation to the results.
            assert np.linalg.det(
                np.cov(
                    prior_result.values.reshape(realizations, NCOL * NROW * NLAY).T[:2]
                )
            ) > np.linalg.det(
                np.cov(
                    posterior_result.values.reshape(realizations, NCOL * NROW * NLAY).T[
                        :2
                    ]
                )
            )

            # 'c' should be inactive (all nans)
            assert np.isnan(
                prior_result.values.reshape(realizations, NCOL * NROW * NLAY).T[2:3]
            ).all()
            assert np.isnan(
                posterior_result.values.reshape(realizations, NCOL * NROW * NLAY).T[2:3]
            ).all()

            # This checks that the fields in the runpath
            # are different between iterations
            assert Path("simulations/realization-0/iter-0/my_param.grdecl").read_text(
                encoding="utf-8"
            ) != Path("simulations/realization-0/iter-1/my_param.grdecl").read_text(
                encoding="utf-8"
            )

            # check shape of written data
            prop0 = xtgeo.gridproperty_from_file(
                "simulations/realization-0/iter-0/my_param.grdecl",
                fformat="grdecl",
                grid=grid,
                name="MY_PARAM",
            )
            assert len(prop0.get_npvalues1d()) == NCOL * NROW * NLAY
            numpy.testing.assert_array_equal(
                np.logical_not(prop0.values1d.mask), mask_list
            )
            assert "nan" not in Path(
                "simulations/realization-0/iter-1/my_param.grdecl"
            ).read_text(encoding="utf-8")


@pytest.mark.timeout(600)
@pytest.mark.filterwarnings("ignore:.*Cross-correlation.*:")
@pytest.mark.filterwarnings("ignore:.*divide by zero.*:")
def test_field_param_update_using_heat_equation_zero_var_params_and_adaptive_loc(
    heat_equation_storage,
):
    """Test field parameter updates with zero-variance regions and adaptive localization.

    This test verifies the behavior of the ensemble smoother update when dealing with
    field parameters that contain regions of zero variance (constant values across all
    realizations). Such scenarios have been reported to cause performance issues and
    numerical instabilities.

    Specifically, this test:
    1. Creates a field where the first 5 layers are set to constant values (1.0)
    2. Performs a smoother update with adaptive localization
    3. Verifies expected numerical warnings are raised due to zero variance
    4. Confirms the update still reduces overall parameter uncertainty

    The test documents known limitations with adaptive localization when handling
    zero-variance regions, particularly:
    - Runtime degradation
    - Numerical warnings from division by zero
    - Cross-correlation matrix instabilities
    """
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="w") as storage:
        experiment = storage.get_experiment_by_name("es-mda")
        prior = experiment.get_ensemble_by_name("default_0")
        cond = prior.load_parameters("COND")

        new_experiment = storage.create_experiment(
            parameters=config.ensemble_config.parameter_configuration,
            responses=config.ensemble_config.response_configuration,
            observations=config.observations,
            name="exp-zero-var",
        )
        new_prior = storage.create_ensemble(
            new_experiment,
            ensemble_size=prior.ensemble_size,
            iteration=0,
            name="prior-zero-var",
        )
        cond["values"][:, :, :5, 0] = 1.0
        for real in range(prior.ensemble_size):
            new_prior.save_parameters("COND", real, cond)

        # Copy responses from existing prior to new prior.
        # Note that we ideally should generate new responses by running the
        # heat equation with the modified prior where parts of the field
        # are given a constant value.
        responses = prior.load_responses("gen_data", range(prior.ensemble_size))
        for realization in range(prior.ensemble_size):
            df = responses.filter(pl.col("realization") == realization)
            new_prior.save_response("gen_data", df, realization)

        new_posterior = storage.create_ensemble(
            new_experiment,
            ensemble_size=config.model_config.num_realizations,
            iteration=1,
            name="new_ensemble",
            prior_ensemble=new_prior,
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")  # Ensure all warnings are always recorded
            smoother_update(
                new_prior,
                new_posterior,
                experiment.observation_keys,
                config.ensemble_config.parameters,
                UpdateSettings(),
                ESSettings(localization=True),
            )

            warning_messages = [(w.category, str(w.message)) for w in record]

            # Check that each required warning appears at least once
            assert any(
                issubclass(w[0], RuntimeWarning)
                and "divide by zero encountered in divide" in w[1]
                for w in warning_messages
            )
            assert any(
                issubclass(w[0], UserWarning)
                and "Cross-correlation matrix has entries not in [-1, 1]" in w[1]
                for w in warning_messages
            )

        param_config = config.ensemble_config.parameter_configs["COND"]
        prior_result = new_prior.load_parameters("COND")["values"]
        posterior_result = new_posterior.load_parameters("COND")["values"]
        prior_covariance = np.cov(
            prior_result.values.reshape(
                new_prior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ).T
        )
        posterior_covariance = np.cov(
            posterior_result.values.reshape(
                new_posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ).T
        )
        # Check that generalized variance is reduced by update step.
        assert np.trace(prior_covariance) > np.trace(posterior_covariance)


@pytest.mark.usefixtures("copy_heat_equation")
def test_foward_init_false():
    config = ErtConfig.from_file("config_forward_init_false.ert")
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "config_forward_init_false.ert",
        "--experiment-name",
        "es-test",
    )

    with open_storage(config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("es-test")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")

        param_config = config.ensemble_config.parameter_configs["COND"]

        prior_result = prior.load_parameters("COND")["values"]
        prior_covariance = np.cov(
            prior_result.values.reshape(
                prior.ensemble_size, param_config.nx * param_config.ny * param_config.nz
            ),
            rowvar=False,
        )

        posterior_result = posterior.load_parameters("COND")["values"]
        posterior_covariance = np.cov(
            posterior_result.values.reshape(
                posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )

        # Check that generalized variance is reduced by update step.
        assert np.trace(prior_covariance) > np.trace(posterior_covariance)
