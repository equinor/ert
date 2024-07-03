import functools
import re
from contextlib import ExitStack as does_not_raise
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
import xarray as xr
import xtgeo
from iterative_ensemble_smoother import steplength_exponential
from scipy.ndimage import gaussian_filter
from tabulate import tabulate

from ert.analysis import (
    ErtAnalysisError,
    ObservationStatus,
    iterative_smoother_update,
    smoother_update,
)
from ert.analysis._es_update import (
    _load_param_ensemble_array,
    _save_param_ensemble_array_to_disk,
)
from ert.analysis.event import AnalysisCompleteEvent, AnalysisErrorEvent
from ert.config import Field, GenDataConfig, GenKwConfig
from ert.config.analysis_config import UpdateSettings
from ert.config.analysis_module import ESSettings, IESSettings
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.field_utils import Shape


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


def remove_timestamp_from_logfile(log_file: Path):
    with open(log_file, "r", encoding="utf-8") as fin:
        buf = fin.read()
    buf = re.sub(
        r"Time: [0-9]{4}\.[0-9]{2}\.[0-9]{2} [0-9]{2}\:[0-9]{2}\:[0-9]{2}", "Time:", buf
    )
    with open(log_file, "w", encoding="utf-8") as fout:
        fout.write(buf)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "misfit_preprocess", [[["*"]], [], [["FOPR"]], [["FOPR"], ["WOPR_OP1_1*"]]]
)
def test_update_report(
    snake_oil_case_storage,
    snake_oil_storage,
    misfit_preprocess,
    snapshot,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    smoother_update(
        prior_ens,
        posterior_ens,
        list(ert_config.observations.keys()),
        ert_config.ensemble_config.parameters,
        UpdateSettings(auto_scale_observations=misfit_preprocess),
        ESSettings(inversion="subspace"),
        progress_callback=events.append,
    )

    event = next(e for e in events if isinstance(e, AnalysisCompleteEvent))
    snapshot.assert_match(
        tabulate(event.data.data, floatfmt=".3f") + "\n", "update_log"
    )


def test_update_report_with_exception_in_analysis_ES(
    snapshot,
    snake_oil_case_storage,
    snake_oil_storage,
):
    ert_config = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    with pytest.raises(
        ErtAnalysisError, match="No active observations for update step"
    ):
        smoother_update(
            prior_ens,
            posterior_ens,
            list(ert_config.observations.keys()),
            ert_config.ensemble_config.parameters,
            UpdateSettings(alpha=0.0000000001),
            ESSettings(inversion="subspace"),
            progress_callback=events.append,
        )

    error_event = next(e for e in events if isinstance(e, AnalysisErrorEvent))
    assert error_event.error_msg == "No active observations for update step"
    snapshot.assert_match(
        tabulate(error_event.data.data, floatfmt=".3f") + "\n", "error_event"
    )


@pytest.mark.parametrize(
    "update_settings",
    [
        UpdateSettings(alpha=0.1),
        UpdateSettings(std_cutoff=0.1),
        UpdateSettings(alpha=0.1, std_cutoff=0.1),
    ],
)
def test_update_report_with_different_observation_status_from_smoother_update(
    update_settings,
    snake_oil_case_storage,
    snake_oil_storage,
):
    ert_config = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")

    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    ss = smoother_update(
        prior_ens,
        posterior_ens,
        list(ert_config.observations.keys()),
        ert_config.ensemble_config.parameters,
        update_settings,
        ESSettings(inversion="subspace"),
        progress_callback=events.append,
    )

    outliers = len(
        [e for e in ss.update_step_snapshots if e.status == ObservationStatus.OUTLIER]
    )
    std_cutoff = len(
        [
            e
            for e in ss.update_step_snapshots
            if e.status == ObservationStatus.STD_CUTOFF
        ]
    )
    missing = len(
        [
            e
            for e in ss.update_step_snapshots
            if e.status == ObservationStatus.MISSING_RESPONSE
        ]
    )
    active = len(ss.update_step_snapshots) - outliers - std_cutoff - missing

    update_event = next(e for e in events if isinstance(e, AnalysisCompleteEvent))
    data_section = update_event.data
    assert data_section.extra["Active observations"] == str(active)
    assert data_section.extra["Deactivated observations - missing respons(es)"] == str(
        missing
    )
    assert data_section.extra[
        "Deactivated observations - ensemble_std > STD_CUTOFF"
    ] == str(std_cutoff)
    assert data_section.extra["Deactivated observations - outliers"] == str(outliers)


@pytest.mark.parametrize(
    "module, expected_gen_kw",
    [
        (
            "IES_ENKF",
            [
                0.5167529669896218,
                -0.9178847938402281,
                -0.6299046429604261,
                -0.1632005925319205,
                0.0216488942750398,
                0.07464619425897459,
                -1.5587692532545538,
                0.22910522740018124,
                -0.7171489000139469,
                0.7287252249699406,
            ],
        ),
        (
            "STD_ENKF",
            [
                1.3040645145742686,
                -0.8162878122658299,
                -1.5484856041224397,
                -1.379896334985399,
                -0.510970027650022,
                0.5638868158813687,
                -2.7669280724377487,
                1.7160680670028017,
                -1.2603717378211836,
                1.2014197463741136,
            ],
        ),
    ],
)
def test_update_snapshot(
    snake_oil_case_storage,
    snake_oil_storage,
    module,
    expected_gen_kw,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage

    # Making sure that row scaling with a row scaling factor of 1.0
    # results in the same update as with ES.
    # Note: seed must be the same!
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )

    # Make sure we always have the same seed in updates
    rng = np.random.default_rng(42)

    if module == "IES_ENKF":
        # Step length defined as a callable on sies-iterations
        sies_step_length = functools.partial(steplength_exponential)

        # The sies-smoother is initially optional
        sies_smoother = None

        # The initial_mask equals ens_mask on first iteration
        initial_mask = prior_ens.get_realization_mask_with_responses()

        # Call an iteration of SIES algorithm. Producing snapshot and SIES obj
        iterative_smoother_update(
            prior_storage=prior_ens,
            posterior_storage=posterior_ens,
            sies_smoother=sies_smoother,
            observations=list(ert_config.observations.keys()),
            parameters=list(ert_config.ensemble_config.parameters),
            update_settings=UpdateSettings(),
            analysis_config=IESSettings(inversion="subspace_exact"),
            sies_step_length=sies_step_length,
            initial_mask=initial_mask,
            rng=rng,
        )
    else:
        smoother_update(
            prior_ens,
            posterior_ens,
            list(ert_config.observations.keys()),
            list(ert_config.ensemble_config.parameters),
            UpdateSettings(),
            ESSettings(inversion="subspace"),
            rng=rng,
        )

    sim_gen_kw = list(
        prior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    target_gen_kw = list(
        posterior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    # Check that prior is not equal to posterior after updationg
    assert sim_gen_kw != target_gen_kw

    # Check that posterior is as expected
    assert target_gen_kw == pytest.approx(expected_gen_kw)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "alpha, expected, expectation",
    [
        pytest.param(
            0.001,
            [],
            pytest.raises(ErtAnalysisError),
            id="Low alpha, no active observations",
        ),
        (
            0.1,
            [
                ObservationStatus.OUTLIER,
                ObservationStatus.OUTLIER,
                ObservationStatus.ACTIVE,
            ],
            does_not_raise(),
        ),
        (
            0.5,
            [
                ObservationStatus.OUTLIER,
                ObservationStatus.ACTIVE,
                ObservationStatus.ACTIVE,
            ],
            does_not_raise(),
        ),
        (
            1,
            [
                ObservationStatus.ACTIVE,
                ObservationStatus.ACTIVE,
                ObservationStatus.ACTIVE,
            ],
            does_not_raise(),
        ),
    ],
)
def test_smoother_snapshot_alpha(
    alpha, expected, storage, uniform_parameter, obs, expectation
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """

    # alpha is a parameter used for outlier detection

    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior_storage = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior_storage.ensemble_size):
        data = rng.uniform(0, 1)
        prior_storage.save_parameters(
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
        prior_storage.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_storage = storage.create_ensemble(
        prior_storage.experiment_id,
        ensemble_size=prior_storage.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_storage,
    )

    # Step length defined as a callable on sies-iterations
    sies_step_length = functools.partial(steplength_exponential)

    # The sies-smoother is initially optional
    sies_smoother = None

    # The initial_mask equals ens_mask on first iteration
    initial_mask = prior_storage.get_realization_mask_with_responses()

    with expectation:
        result_snapshot, _ = iterative_smoother_update(
            prior_storage=prior_storage,
            posterior_storage=posterior_storage,
            sies_smoother=sies_smoother,
            observations=["OBSERVATION"],
            parameters=["PARAMETER"],
            update_settings=UpdateSettings(alpha=alpha),
            analysis_config=IESSettings(),
            sies_step_length=sies_step_length,
            initial_mask=initial_mask,
        )
        assert result_snapshot.alpha == alpha
        assert [
            step.status for step in result_snapshot.update_step_snapshots
        ] == expected


def test_and_benchmark_adaptive_localization_with_fields(
    storage, tmp_path, monkeypatch, benchmark
):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)

    num_grid_cells = 1000
    num_parameters = num_grid_cells * num_grid_cells
    num_observations = 50
    num_ensemble = 25

    # Create a tridiagonal matrix that maps responses to parameters.
    # Being tridiagonal, it ensures that each response is influenced only by its neighboring parameters.
    diagonal = np.ones(min(num_parameters, num_observations))
    A = sp.sparse.diags(
        [diagonal, diagonal, diagonal],
        offsets=[-1, 0, 1],
        shape=(num_observations, num_parameters),
        dtype=float,
    ).toarray()

    # We add some noise that is insignificant compared to the
    # actual local structure in the forward model step
    A += rng.standard_normal(size=A.shape) * 0.01

    def g(X):
        """Apply the forward model."""
        return A @ X

    all_realizations = np.zeros((num_ensemble, num_grid_cells, num_grid_cells, 1))

    # Generate num_ensemble realizations of the Gaussian Random Field
    for i in range(num_ensemble):
        sigma = 10
        realization = np.exp(
            gaussian_filter(
                gaussian_filter(
                    rng.standard_normal((num_grid_cells, num_grid_cells)), sigma=sigma
                ),
                sigma=sigma,
            )
        )

        realization = realization[..., np.newaxis]
        all_realizations[i] = realization

    X = all_realizations.reshape(-1, num_grid_cells * num_grid_cells).T

    Y = g(X)

    # Create observations by adding noise to a realization.
    observation_noise = rng.standard_normal(size=num_observations)
    observations = Y[:, 0] + observation_noise

    # Create necessary files and data sets to be able to update
    # the parameters using the ensemble smoother.
    shape = Shape(num_grid_cells, num_grid_cells, 1)
    grid = xtgeo.create_box_grid(dimension=(shape.nx, shape.ny, shape.nz))
    grid.to_file("MY_EGRID.EGRID", "egrid")

    resp = GenDataConfig(name="RESPONSE")
    obs = xr.Dataset(
        {
            "observations": (
                ["report_step", "index"],
                observations.reshape((1, num_observations)),
            ),
            "std": (
                ["report_step", "index"],
                observation_noise.reshape(1, num_observations),
            ),
        },
        coords={"report_step": [0], "index": np.arange(len(observations))},
        attrs={"response": "RESPONSE"},
    )

    param_group = "PARAM_FIELD"

    config = Field.from_config_list(
        "MY_EGRID.EGRID",
        shape,
        [
            param_group,
            param_group,
            "param.GRDECL",
            "INIT_FILES:param_%d.GRDECL",
            "FORWARD_INIT:False",
        ],
    )

    experiment = storage.create_experiment(
        parameters=[config],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )

    prior_ensemble = storage.create_ensemble(
        experiment,
        ensemble_size=num_ensemble,
        iteration=0,
        name="prior",
    )

    for iens in range(prior_ensemble.ensemble_size):
        prior_ensemble.save_parameters(
            param_group,
            iens,
            xr.Dataset(
                {
                    "values": xr.DataArray(
                        X[:, iens].reshape(num_grid_cells, num_grid_cells, 1),
                        dims=("x", "y", "z"),
                    ),
                }
            ),
        )

        prior_ensemble.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [Y[:, iens]])},
                coords={"index": range(len(Y[:, iens])), "report_step": [0]},
            ),
            iens,
        )

    posterior_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=prior_ensemble.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ensemble,
    )

    smoother_update_run = partial(
        smoother_update,
        prior_ensemble,
        posterior_ensemble,
        ["OBSERVATION"],
        [param_group],
        UpdateSettings(),
        ESSettings(localization=True),
    )
    benchmark(smoother_update_run)

    prior_da = prior_ensemble.load_parameters(param_group, range(num_ensemble))[
        "values"
    ]
    posterior_da = posterior_ensemble.load_parameters(param_group, range(num_ensemble))[
        "values"
    ]
    # Make sure some, but not all parameters were updated.
    assert not np.allclose(prior_da, posterior_da)
    # All parameters would be updated with a global update so this would fail.
    assert np.isclose(prior_da, posterior_da).sum() > 0
    # The std for the ensemble should decrease
    assert float(
        prior_ensemble.calculate_std_dev_for_parameter(param_group)["values"].sum()
    ) > float(
        posterior_ensemble.calculate_std_dev_for_parameter(param_group)["values"].sum()
    )


def test_update_only_using_subset_observations(
    snake_oil_case_storage, snake_oil_storage, snapshot
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage

    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    smoother_update(
        prior_ens,
        posterior_ens,
        ["WPR_DIFF_1"],
        ert_config.ensemble_config.parameters,
        UpdateSettings(),
        ESSettings(),
        progress_callback=events.append,
    )

    update_event = next(e for e in events if isinstance(e, AnalysisCompleteEvent))
    snapshot.assert_match(
        tabulate(update_event.data.data, floatfmt=".3f") + "\n", "update_log"
    )


def test_temporary_parameter_storage_with_inactive_fields(
    storage, tmp_path, monkeypatch
):
    """
    Tests that when FIELDS with inactive cells are stored in the temporary
    parameter storage the inactive cells are not stored along with the active cells.

    Then test that we restore the inactive cells when saving the temporary
    parameter storage to disk again.
    """
    monkeypatch.chdir(tmp_path)

    num_grid_cells = 40
    layers = 5
    ensemble_size = 5
    param_group = "PARAM_FIELD"
    shape = Shape(num_grid_cells, num_grid_cells, layers)

    grid = xtgeo.create_box_grid(dimension=(shape.nx, shape.ny, shape.nz))
    mask = grid.get_actnum()
    rng = np.random.default_rng()
    mask_list = rng.choice([True, False], shape.nx * shape.ny * shape.nz)
    mask.values = mask_list
    grid.set_actnum(mask)
    grid.to_file("MY_EGRID.EGRID", "egrid")

    config = Field.from_config_list(
        "MY_EGRID.EGRID",
        shape,
        [
            param_group,
            param_group,
            "param.GRDECL",
            "INIT_FILES:param_%d.GRDECL",
            "FORWARD_INIT:False",
        ],
    )

    experiment = storage.create_experiment(
        parameters=[config],
        name="my_experiment",
    )

    prior_ensemble = storage.create_ensemble(
        experiment=experiment,
        ensemble_size=ensemble_size,
        iteration=0,
        name="prior",
    )
    fields = [
        xr.Dataset(
            {
                "values": (
                    ["x", "y", "z"],
                    np.ma.MaskedArray(
                        data=rng.random(size=(shape.nx, shape.ny, shape.nz)),
                        fill_value=np.nan,
                        mask=[~mask_list],
                    ).filled(),
                )
            }
        )
        for _ in range(ensemble_size)
    ]

    for iens in range(ensemble_size):
        prior_ensemble.save_parameters(param_group, iens, fields[iens])

    realization_list = list(range(ensemble_size))
    param_ensemble_array = _load_param_ensemble_array(
        prior_ensemble, param_group, realization_list
    )

    assert np.count_nonzero(mask_list) < (shape.nx * shape.ny * shape.nz)
    assert param_ensemble_array.shape == (
        np.count_nonzero(mask_list),
        ensemble_size,
    )

    ensemble = storage.create_ensemble(
        experiment=experiment,
        ensemble_size=ensemble_size,
        iteration=0,
        name="post",
    )

    _save_param_ensemble_array_to_disk(
        ensemble, param_ensemble_array, param_group, realization_list
    )
    for iens in range(prior_ensemble.ensemble_size):
        ds = xr.open_dataset(
            ensemble._path / f"realization-{iens}" / f"{param_group}.nc", engine="scipy"
        )
        np.testing.assert_array_equal(ds["values"].values[0], fields[iens]["values"])


def test_that_observations_keep_sorting(snake_oil_case_storage, snake_oil_storage):
    """
    The order of the observations influence the update as it affects the
    perturbations, so we make sure we maintain the order throughout.
    """
    ert_config = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    assert list(ert_config.observations.keys()) == list(
        prior_ens.experiment.observations.keys()
    )
