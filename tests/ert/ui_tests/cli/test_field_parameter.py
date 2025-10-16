import logging
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
from polars import Float32

from ert.analysis import smoother_update
from ert.config import ErtConfig, ESSettings, ObservationSettings
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli


def test_field_param_update_using_heat_equation_enif(
    symlinked_heat_equation_storage_enif,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        experiment = storage.get_experiment_by_name("enif")
        [prior, posterior] = experiment.ensembles

        prior_result = prior.load_parameters("COND")["values"]

        param_config = config.ensemble_config.parameter_configs["COND"]
        assert len(prior_result.x) == param_config.nx
        assert len(prior_result.y) == param_config.ny
        assert len(prior_result.z) == param_config.nz

        posterior_result = posterior.load_parameters("COND")["values"]
        prior_covariance = np.cov(
            prior_result.values.reshape(
                prior.ensemble_size, param_config.nx * param_config.ny * param_config.nz
            ),
            rowvar=False,
        )
        posterior_covariance = np.cov(
            posterior_result.values.reshape(
                posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )
        # Check that generalized variance is reduced by update step.
        assert np.trace(prior_covariance) > np.trace(posterior_covariance)

    # Check that fields in the runpath are different between iterations
    cond_iter0 = resfo.read("simulations/realization-0/iter-0/cond.bgrdecl")[0][1]
    cond_iter1 = resfo.read("simulations/realization-0/iter-1/cond.bgrdecl")[0][1]
    assert (cond_iter0 != cond_iter1).all()


def _compare_ensemble_params(
    actual: pl.DataFrame,
    reference_path: Path,
    index_columns: list[str],
    outlier_threshold: float = 1e-6,
    outlier_percentage_max: float = 0.05,
    outlier_deviation_max: float = 1e-1,
    update_snapshot: bool = False,
) -> pl.DataFrame:
    if not reference_path.exists():
        if update_snapshot:
            actual.write_csv(reference_path)
            raise AssertionError(f"Snapshot @ {reference_path} changed")
        else:
            raise AssertionError(
                f"No snapshot @ {reference_path} found. "
                "Run with --snapshot-update to create a new snapshot."
            )

    expected = pl.read_csv(reference_path)
    if actual.shape != expected.shape:
        raise ValueError("DataFrames must have the same shape")

    actual_numbers = actual.drop(index_columns)
    expected_numbers = expected.drop(index_columns)
    columns = actual_numbers.columns

    diff_df = actual_numbers - expected_numbers

    # Truncate small differences to zero
    truncated_df = diff_df.select(
        [
            (
                pl.col(c).map_elements(
                    lambda x: 0.0
                    if abs(x) < outlier_threshold
                    else (abs(x) - outlier_threshold),
                    return_dtype=Float32,
                )
            ).alias(c)
            for c in columns
        ]
    )

    # Compute per-row percentage of non-zero (i.e., outlier) values
    outlier_percentage = truncated_df.select(
        [
            pl.concat_list(columns)
            .map_elements(
                lambda row: sum(1 for x in row if x != 0.0) / len(row),
                return_dtype=Float32,
            )
            .alias("outlier_percentage")
        ]
    )["outlier_percentage"]

    max_deviance = truncated_df.select(
        [
            pl.concat_list(columns)
            .map_elements(lambda row: max(x for x in row), return_dtype=Float32)
            .alias("max_deviance")
        ]
    )["max_deviance"]

    unacceptable_n_outliers = (outlier_percentage > outlier_percentage_max).any()
    unacceptable_outlier_deviation = (max_deviance > outlier_deviation_max).any()

    _schema = expected.schema

    def round_and_cast_(df: pl.DataFrame) -> pl.DataFrame:
        return df.cast(_schema).with_columns(
            pl.col(pl.Float64).round(int(1 / outlier_threshold))
        )

    if update_snapshot and not round_and_cast_(actual).equals(
        round_and_cast_(expected)
    ):
        actual.write_csv(reference_path)
        raise AssertionError(f"Snapshot @ {reference_path} changed!")

    if unacceptable_n_outliers or unacceptable_outlier_deviation:
        summary_table = actual.with_columns(outlier_percentage, max_deviance)
        realizations = sorted(map(int, set(actual.columns) - set(index_columns)))

        summary_str = "\n".join(
            [
                "".join(
                    [f"{i}: ".rjust(6)]
                    + ["x" if row[str(r)] > 0 else " " for r in realizations]
                    + [f"  outliers: {100 * row['outlier_percentage']:.1f}%, ".rjust(4)]
                    + [f"max deviance={row['max_deviance']:.2e}"]
                )
                for i, row in enumerate(
                    summary_table.filter(
                        (outlier_percentage > outlier_percentage_max)
                        | (max_deviance > outlier_deviation_max)
                    ).to_dicts()
                )
            ]
        )

        raise AssertionError(
            f"Changes in snapshot detected.\n"
            f"Max row-wise outlier percentage: {max(outlier_percentage) * 100}%\n"
            f"Max param deviation from reference: {max(max_deviance)}\n"
            f":\n{summary_str}"
        )


def test_field_param_update_using_heat_equation_enif_snapshot(
    symlinked_heat_equation_storage_enif, snapshot, request
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        experiment = storage.get_experiment_by_name("enif")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")

        prior_result = prior.load_parameters("COND")["values"]

        param_config = config.ensemble_config.parameter_configs["COND"]
        assert len(prior_result.x) == param_config.nx
        assert len(prior_result.y) == param_config.ny
        assert len(prior_result.z) == param_config.nz

        data = []
        for i, ens in enumerate([prior, posterior]):
            field_data = ens.load_parameters("COND")
            field_df = pl.from_pandas(
                field_data.to_dataframe().reset_index()
            ).with_columns(pl.lit(i).alias("iteration"))
            field_df = field_df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            data.append(
                field_df.select("iteration", "realizations", "x", "y", "z", "values")
            )

        result = pl.concat(data)
        result = result.sort(["iteration", "realizations", "x", "y", "z"])
        result = result.pivot(on=["realizations"], values="values", sort_columns=True)

        _compare_ensemble_params(
            actual=result,
            reference_path=snapshot.snapshot_dir / "enif_heat_snapshot.csv",
            outlier_threshold=0.01,
            index_columns=result.columns[:4],
            outlier_percentage_max=0.03,
            outlier_deviation_max=0.1,
            update_snapshot=bool(request.config.getoption("--snapshot-update")),
        )


def test_field_param_update_using_heat_equation(symlinked_heat_equation_storage_es):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        experiment = storage.get_experiment_by_name("es")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")

        prior_result = prior.load_parameters("COND")["values"]

        param_config = config.ensemble_config.parameter_configs["COND"]
        assert len(prior_result.x) == param_config.nx
        assert len(prior_result.y) == param_config.ny
        assert len(prior_result.z) == param_config.nz

        posterior_result = posterior.load_parameters("COND")["values"]
        assert posterior_result.dtype == np.float32
        prior_covariance = np.cov(
            prior_result.values.reshape(
                prior.ensemble_size, param_config.nx * param_config.ny * param_config.nz
            ),
            rowvar=False,
        )
        posterior_covariance = np.cov(
            posterior_result.values.reshape(
                posterior.ensemble_size,
                param_config.nx * param_config.ny * param_config.nz,
            ),
            rowvar=False,
        )
        # Check that generalized variance is reduced by update step.
        assert np.trace(prior_covariance) > np.trace(posterior_covariance)

    # Check that fields in the runpath are different between iterations
    cond_iter0 = resfo.read("simulations/realization-0/iter-0/cond.bgrdecl")[0][1]
    cond_iter1 = resfo.read("simulations/realization-0/iter-1/cond.bgrdecl")[0][1]
    assert (cond_iter0 != cond_iter1).all()


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_field_parameter_persistence_to_grdecl(tmpdir):
    """
    This replicates the poly example, only it uses FIELD parameter
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
            NUM_REALIZATIONS 5
            OBS_CONFIG observations
            FIELD MY_PARAM PARAMETER my_param.grdecl \
                INIT_FILES:my_param.grdecl FORWARD_INIT:True
            GRID MY_EGRID.EGRID
            GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out \
                REPORT_STEPS:0
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

        grid.to_file("MY_EGRID.EGRID", "egrid")

        Path("forward_model").write_text(
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
            ),
            encoding="utf-8",
        )

        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        Path("POLY_EVAL").write_text("EXECUTABLE forward_model", encoding="utf-8")
        Path("observations").write_text(
            dedent(
                """
            GENERAL_OBSERVATION MY_OBS {
                DATA       = MY_RESPONSE;
                INDEX_LIST = 0,2,4,6,8;
                RESTART    = 0;
                OBS_FILE   = obs.txt;
            };"""
            ),
            encoding="utf-8",
        )

        Path("obs.txt").write_text(
            dedent(
                """
            2.1457049781272213 0.6
            8.769219841380755 1.4
            12.388014786122742 3.0
            25.600464531354252 5.4
            42.35204755970952 8.6"""
            ),
            encoding="utf-8",
        )

        run_cli(
            ENSEMBLE_SMOOTHER_MODE,
            "--disable-monitoring",
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
            assert "nan" not in Path(
                "simulations/realization-0/iter-1/my_param.grdecl"
            ).read_text(encoding="utf-8")


@pytest.mark.timeout(600)
def test_field_param_update_using_heat_equation_zero_var_params_and_adaptive_loc(
    symlinked_heat_equation_storage_es, caplog
):
    """Test field parameter updates with zero-variance regions and adaptive
    localization.

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
        experiment = storage.get_experiment_by_name("es")
        prior = experiment.get_ensemble_by_name("iter-0")
        cond = prior.load_parameters("COND")
        init_temp_scale = prior.load_parameters("INIT_TEMP_SCALE")
        corr_length = prior.load_parameters("CORR_LENGTH")

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
            new_prior.save_parameters(cond, "COND", real)
            new_prior.save_parameters(init_temp_scale, "INIT_TEMP_SCALE", real)
            new_prior.save_parameters(corr_length, "CORR_LENGTH", real)

        # Copy responses from existing prior to new prior.
        # Note that we ideally should generate new responses by running the
        # heat equation with the modified prior where parts of the field
        # are given a constant value.
        responses = prior.load_responses("gen_data", tuple(range(prior.ensemble_size)))
        for realization in range(prior.ensemble_size):
            df = responses.filter(pl.col("realization") == realization)
            new_prior.save_response("gen_data", df, realization)

        new_posterior = storage.create_ensemble(
            new_experiment,
            ensemble_size=config.runpath_config.num_realizations,
            iteration=1,
            name="new_ensemble",
            prior_ensemble=new_prior,
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")  # Ensure all warnings are always recorded
            with caplog.at_level(logging.INFO):
                smoother_update(
                    new_prior,
                    new_posterior,
                    experiment.observation_keys,
                    config.ensemble_config.parameters,
                    ObservationSettings(),
                    ESSettings(localization=True),
                )

                # Note that this used to fail since run time and user warnings were
                # thrown because we tried updating parameters with zero variance.
                assert not record

                assert (
                    "There are 50 parameters with 0 variance that will not be updated."
                    in caplog.text
                )

        param_config = config.ensemble_config.parameter_configs["COND"]
        prior_result = new_prior.load_parameters("COND")["values"]
        posterior_result = new_posterior.load_parameters("COND")["values"]

        # Make sure parameters with zero variance are not updated.
        assert (
            prior_result.values[:, :, :5, 0] == posterior_result.values[:, :, :5, 0]
        ).all()

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
        "--disable-monitoring",
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
