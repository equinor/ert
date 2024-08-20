import os
import stat
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter
from xtgeo import RegularSurface, surface_from_file

from ert import LibresFacade
from ert.analysis import ErtAnalysisError, ObservationStatus, smoother_update
from ert.analysis._es_update import _all_parameters
from ert.analysis.event import AnalysisCompleteEvent
from ert.config import ErtConfig, GenDataConfig, GenKwConfig
from ert.config.analysis_config import UpdateSettings
from ert.config.analysis_module import ESSettings
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from ert.storage.realization_storage_state import RealizationStorageState
from tests.integration_tests.run_cli import run_cli


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
        ],
        output_file="kw.txt",
        update=True,
    )


@pytest.fixture
def obs():
    return xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0, 1.0, 1.0]]),
            "std": (["report_step", "index"], [[0.1, 1.0, 10.0]]),
        },
        coords={"index": [0, 1, 2], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_that_posterior_has_lower_variance_than_prior():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "--realizations",
        "1-50",
        "--experiment-name",
        "testing",
        "poly.ert",
    )
    facade = LibresFacade.from_config_file("poly.ert")
    with open_storage(facade.enspath) as storage:
        experiment = storage.get_experiment_by_name("testing")
        prior_ensemble = experiment.get_ensemble_by_name("iter-0")
        df_default = prior_ensemble.load_all_gen_kw_data()
        posterior_ensemble = experiment.get_ensemble_by_name("iter-1")
        df_target = posterior_ensemble.load_all_gen_kw_data()

        # The std for the ensemble should decrease
        assert float(
            prior_ensemble.calculate_std_dev_for_parameter("COEFFS")["values"].sum()
        ) > float(
            posterior_ensemble.calculate_std_dev_for_parameter("COEFFS")["values"].sum()
        )

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_target.cov().to_numpy())
        < np.linalg.det(df_default.cov().to_numpy())
    )


@pytest.mark.integration_test
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
        "--disable-monitor",
        "snake_oil_surface.ert",
    )

    ert_config = ErtConfig.from_file("snake_oil_surface.ert")

    storage = open_storage(ert_config.ens_path)

    ens_prior = storage.get_ensemble_by_name("iter-0")
    ens_posterior = storage.get_ensemble_by_name("iter-1")

    # Check that surfaces defined in INIT_FILES are not changed by ERT
    surf_prior = ens_prior.load_parameters("TOP", list(range(ensemble_size)))["values"]
    for i in range(ensemble_size):
        _prior_init = surface_from_file(
            f"surface/surf_init_{i}.irap", fformat="irap_ascii", dtype=np.float32
        )
        np.testing.assert_array_equal(surf_prior[i], _prior_init.values.data)

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


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_snake_oil_field")
def test_update_multiple_param():
    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitor",
        "snake_oil.ert",
    )

    ert_config = ErtConfig.from_file("snake_oil.ert")

    storage = open_storage(ert_config.ens_path)
    experiment = storage.get_experiment_by_name("ensemble-experiment")
    prior_ensemble = experiment.get_ensemble_by_name("iter-0")
    posterior_ensemble = experiment.get_ensemble_by_name("iter-1")

    prior_array = _all_parameters(prior_ensemble, list(range(10)))
    posterior_array = _all_parameters(posterior_ensemble, list(range(10)))

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    # https://en.wikipedia.org/wiki/Variance#For_vector-valued_random_variables
    assert np.trace(np.cov(posterior_array)) < np.trace(np.cov(prior_array))


@pytest.mark.integration_test
def test_gen_data_obs_data_mismatch(storage, uniform_parameter):
    resp = GenDataConfig(name="RESPONSE")
    obs = xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0]]),
            "std": (["report_step", "index"], [[0.1]]),
        },
        coords={"index": [1000], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        data = rng.uniform(0.8, 1, 3)
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    with pytest.raises(
        ErtAnalysisError,
        match="No active observations",
    ):
        smoother_update(
            prior,
            posterior_ens,
            ["OBSERVATION"],
            ["PARAMETER"],
            UpdateSettings(),
            ESSettings(),
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_gen_data_missing(storage, uniform_parameter, obs):
    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        data = rng.uniform(0.8, 1, 2)  # Importantly, shorter than obs
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    events = []

    update_snapshot = smoother_update(
        prior,
        posterior_ens,
        ["OBSERVATION"],
        ["PARAMETER"],
        UpdateSettings(),
        ESSettings(),
        progress_callback=events.append,
    )
    assert [step.status for step in update_snapshot.update_step_snapshots] == [
        ObservationStatus.ACTIVE,
        ObservationStatus.ACTIVE,
        ObservationStatus.MISSING_RESPONSE,
    ]

    update_event = next(e for e in events if isinstance(e, AnalysisCompleteEvent))
    data_section = update_event.data
    assert data_section.extra["Active observations"] == "2"
    assert data_section.extra["Deactivated observations - missing respons(es)"] == "1"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_update_subset_parameters(storage, uniform_parameter, obs):
    no_update_param = GenKwConfig(
        name="EXTRA_PARAMETER",
        forward_init=False,
        template_file="",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
        ],
        output_file=None,
        update=False,
    )
    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter, no_update_param],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        prior.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        prior.save_parameters(
            "EXTRA_PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )

        data = rng.uniform(0.8, 1, 10)
        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    smoother_update(
        prior,
        posterior_ens,
        ["OBSERVATION"],
        ["PARAMETER"],
        UpdateSettings(),
        ESSettings(),
    )
    assert prior.load_parameters("EXTRA_PARAMETER", 0)["values"].equals(
        posterior_ens.load_parameters("EXTRA_PARAMETER", 0)["values"]
    )
    assert not prior.load_parameters("PARAMETER", 0)["values"].equals(
        posterior_ens.load_parameters("PARAMETER", 0)["values"]
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_that_update_works_with_failed_realizations():
    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import numpy as np
                import sys
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["COEFFS"]

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    if np.random.random(1) > 0.5:
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
        "--disable-monitor",
        "poly.ert",
    )

    ert_config = ErtConfig.from_file("poly.ert")

    with open_storage(ert_config.ens_path) as storage:
        prior = storage.get_ensemble_by_name("iter-0")
        posterior = storage.get_ensemble_by_name("iter-1")

        assert all(
            posterior.get_ensemble_state()[idx]
            == RealizationStorageState.PARENT_FAILURE
            for idx, v in enumerate(prior.get_ensemble_state())
            if v == RealizationStorageState.LOAD_FAILURE
        )
