from contextlib import ExitStack as does_not_raise
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xarray as xr
import xtgeo
from tabulate import tabulate

from ert.analysis import ErtAnalysisError, ObservationStatus, smoother_update
from ert.analysis._update_commons import (
    _compute_observation_statuses,
    _OutlierColumns,
    _preprocess_observations_and_responses,
)
from ert.analysis.event import AnalysisCompleteEvent, AnalysisErrorEvent
from ert.config import (
    ESSettings,
    Field,
    GenDataConfig,
    GenKwConfig,
    ObservationSettings,
    OutlierSettings,
)
from ert.field_utils import Shape
from ert.storage import Ensemble, open_storage


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="KEY_1",
        group="PARAMETER",
        distribution={"name": "uniform", "min": 0, "max": 1},
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


@pytest.mark.integration_test
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
    experiment = snake_oil_storage.get_experiment_by_name("ensemble-experiment")
    prior_ens = experiment.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.runpath_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    smoother_update(
        prior_ens,
        posterior_ens,
        experiment.observation_keys,
        ert_config.ensemble_config.parameters,
        ObservationSettings(auto_scale_observations=misfit_preprocess),
        ESSettings(inversion="SUBSPACE"),
        progress_callback=events.append,
    )

    event = next(e for e in events if isinstance(e, AnalysisCompleteEvent))
    snapshot.assert_match(
        tabulate(event.data.data, floatfmt=".3f") + "\n", "update_log"
    )


@pytest.mark.integration_test
def test_update_report_with_exception_in_analysis_ES(
    snapshot,
    snake_oil_case_storage,
    snake_oil_storage,
):
    ert_config = snake_oil_case_storage
    experiment = snake_oil_storage.get_experiment_by_name("ensemble-experiment")
    prior_ens = experiment.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.runpath_config.num_realizations,
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
            experiment.observation_keys,
            ert_config.ensemble_config.parameters,
            ObservationSettings(outlier_settings=OutlierSettings(alpha=0.0000000001)),
            ESSettings(inversion="SUBSPACE"),
            progress_callback=events.append,
        )

    error_event = next(e for e in events if isinstance(e, AnalysisErrorEvent))
    assert error_event.error_msg == "No active observations for update step"
    snapshot.assert_match(
        tabulate(error_event.data.data, floatfmt=".3f") + "\n", "error_event"
    )


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "update_settings, num_overspread, num_collapsed, num_nan, num_active",
    [
        (
            ObservationSettings(outlier_settings=OutlierSettings(alpha=0.1)),
            169,
            0,
            0,
            41,
        ),
        (
            ObservationSettings(outlier_settings=OutlierSettings(std_cutoff=0.1)),
            0,
            73,
            0,
            137,
        ),
        (
            ObservationSettings(
                outlier_settings=OutlierSettings(alpha=0.1, std_cutoff=0.1)
            ),
            113,
            73,
            0,
            24,
        ),
    ],
)
def test_update_report_with_different_observation_status_from_smoother_update(
    update_settings,
    num_overspread,
    num_collapsed,
    num_nan,
    num_active,
    snake_oil_case_storage,
    snake_oil_storage,
):
    ert_config = snake_oil_case_storage
    experiment = snake_oil_storage.get_experiment_by_name("ensemble-experiment")
    prior_ens = experiment.get_ensemble_by_name("default_0")

    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.runpath_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    events = []

    ss = smoother_update(
        prior_ens,
        posterior_ens,
        experiment.observation_keys,
        ert_config.ensemble_config.parameters,
        update_settings,
        ESSettings(inversion="SUBSPACE"),
        progress_callback=events.append,
    )

    assert (
        num_overspread
        == ss.observations_and_responses.filter(
            pl.col("status") == ObservationStatus.OUTLIER
        ).height
    )
    assert (
        num_collapsed
        == ss.observations_and_responses.filter(
            pl.col("status") == ObservationStatus.STD_CUTOFF
        ).height
    )
    assert (
        num_nan
        == ss.observations_and_responses.filter(
            pl.col("status") == ObservationStatus.MISSING_RESPONSE
        ).height
    )
    assert (
        num_active
        == ss.observations_and_responses.filter(
            pl.col("status") == ObservationStatus.ACTIVE
        ).height
    )


def test_update_handles_precision_loss_in_std_dev(tmp_path):
    """
    This is a regression test for precision loss in calculating
    standard deviation.
    """
    gen_kw = GenKwConfig(
        name="coeff_0",
        group="COEFFS",
        distribution={"name": "const", "value": 0.1},
    )
    # The values given here are chosen so that when computing
    # `ens_std = S.std(ddof=0, axis=1)`, ens_std[0] is not zero even though
    # all responses have the same value: 5.08078746e07.
    # This is due to precision loss.
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="ensemble_smoother",
            parameters=[gen_kw],
            observations={
                "gen_data": pl.DataFrame(
                    {
                        "response_key": "RES",
                        "observation_key": "OBS",
                        "report_step": pl.Series(np.zeros(3), dtype=pl.UInt16),
                        "index": pl.Series([0, 1, 2], dtype=pl.UInt16),
                        "observations": pl.Series(
                            [-218285263.28648496, -999999999.0, -107098474.0148249],
                            dtype=pl.Float32,
                        ),
                        "std": pl.Series(
                            [559437122.6211826, 999999999.9999999, 1.9],
                            dtype=pl.Float32,
                        ),
                    }
                )
            },
            responses=[
                GenDataConfig(
                    name="gen_data",
                    input_files=["poly.out"],
                    keys=["RES"],
                    has_finalized_keys=True,
                    report_steps_list=[None],
                )
            ],
        )
        prior = storage.create_ensemble(experiment.id, ensemble_size=23, name="prior")
        datasets = [
            Ensemble.sample_parameter(
                gen_kw,
                realization_nr,
                random_seed=1234,
            )
            for realization_nr in range(prior.ensemble_size)
        ]
        prior.save_parameters(pl.concat(datasets, how="vertical"))

        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RES",
                    "report_step": pl.Series(np.zeros(3), dtype=pl.UInt16),
                    "index": pl.Series(np.arange(3), dtype=pl.UInt16),
                    "values": pl.Series(
                        np.array([5.08078746e07, 4.07677769e10, 2.28002538e12]),
                        dtype=pl.Float32,
                    ),
                }
            ),
            0,
        )
        for i in range(1, prior.ensemble_size):
            prior.save_response(
                "gen_data",
                pl.DataFrame(
                    {
                        "response_key": "RES",
                        "report_step": pl.Series(np.zeros(3), dtype=pl.UInt16),
                        "index": pl.Series(np.arange(3), dtype=pl.UInt16),
                        "values": pl.Series(
                            np.array([5.08078744e07, 4.12422210e09, 1.26490794e10]),
                            dtype=pl.Float32,
                        ),
                    }
                ),
                i,
            )

        posterior = storage.create_ensemble(
            prior.experiment_id,
            ensemble_size=prior.ensemble_size,
            iteration=1,
            name="posterior",
            prior_ensemble=prior,
        )
        events = []

        ss = smoother_update(
            prior,
            posterior,
            experiment.observation_keys,
            ["coeff_0"],
            ObservationSettings(auto_scale_observations=[["OBS*"]]),
            ESSettings(),
            progress_callback=events.append,
        )

        assert (
            ss.observations_and_responses.filter(
                pl.col("status") == ObservationStatus.STD_CUTOFF
            ).height
            == 1
        )


def test_update_raises_on_singular_matrix(tmp_path):
    """
    This is a regression test for precision loss in calculating
    standard deviation.
    """
    gen_kw = GenKwConfig(
        name="coeff_0",
        group="COEFFS",
        distribution={"name": "const", "value": 0.1},
    )
    # The values given here are chosen so that when computing
    # `ens_std = S.std(ddof=0, axis=1)`, ens_std[0] is not zero even though
    # all responses have the same value: 5.08078746e07.
    # This is due to precision loss.
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="ensemble_smoother",
            parameters=[gen_kw],
            observations={
                "gen_data": pl.DataFrame(
                    {
                        "response_key": "RES",
                        "observation_key": "OBS",
                        "report_step": pl.Series(np.zeros(3), dtype=pl.UInt16),
                        "index": pl.Series([0, 1, 2], dtype=pl.UInt16),
                        "observations": pl.Series(
                            [-1.5, 5.9604645e-08, 0.0],
                            dtype=pl.Float32,
                        ),
                        "std": pl.Series(
                            [0.33333334, 0.14142136, 0.0],
                            dtype=pl.Float32,
                        ),
                    }
                )
            },
            responses=[
                GenDataConfig(
                    name="gen_data",
                    input_files=["poly.out"],
                    keys=["RES"],
                    has_finalized_keys=True,
                    report_steps_list=[None],
                )
            ],
        )
        prior = storage.create_ensemble(experiment.id, ensemble_size=2, name="prior")
        datasets = [
            Ensemble.sample_parameter(
                gen_kw,
                realization_nr,
                random_seed=1234,
            )
            for realization_nr in range(prior.ensemble_size)
        ]
        prior.save_parameters(pl.concat(datasets, how="vertical"))

        for i, v in enumerate(
            [
                [5.4112810e-01, 2.7799776e08, 1.0093105e10],
                [4.1801736e-01, 5.9196467e08, 2.2322526e10],
            ]
        ):
            prior.save_response(
                "gen_data",
                pl.DataFrame(
                    {
                        "response_key": "RES",
                        "report_step": pl.Series(np.zeros(3), dtype=pl.UInt16),
                        "index": pl.Series(np.arange(3), dtype=pl.UInt16),
                        "values": pl.Series(
                            np.array(v),
                            dtype=pl.Float32,
                        ),
                    }
                ),
                i,
            )

        posterior = storage.create_ensemble(
            prior.experiment_id,
            ensemble_size=prior.ensemble_size,
            iteration=1,
            name="posterior",
            prior_ensemble=prior,
        )

        with (
            pytest.raises(
                ErtAnalysisError,
                match=r"Failed while computing transition matrix.* Matrix is singular",
            ),
            pytest.warns(RuntimeWarning, match="divide by zero"),
        ):
            _ = smoother_update(
                prior,
                posterior,
                experiment.observation_keys,
                ["coeff_0"],
                ObservationSettings(auto_scale_observations=[["OBS*"]]),
                ESSettings(),
                rng=np.random.default_rng(1234),
            )


@pytest.mark.integration_test
def test_update_snapshot(
    snake_oil_case_storage,
    snake_oil_storage,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    expected_gen_kw = [
        1.7365584618531105,
        -0.819068074727709,
        -1.6628460358849138,
        -1.269803440396085,
        -0.06688718485326725,
        0.5544021609832737,
        -2.904293766981197,
        1.6866443742416257,
        -1.6783511959093573,
        1.3081213916230614,
    ]
    ert_config = snake_oil_case_storage

    # Making sure that row scaling with a row scaling factor of 1.0
    # results in the same update as with ES.
    # Note: seed must be the same!
    experiment = snake_oil_storage.get_experiment_by_name("ensemble-experiment")
    prior_ens = experiment.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.runpath_config.num_realizations,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )

    # Make sure we always have the same seed in updates
    rng = np.random.default_rng(42)

    smoother_update(
        prior_ens,
        posterior_ens,
        experiment.observation_keys,
        list(ert_config.ensemble_config.parameters),
        ObservationSettings(),
        ESSettings(inversion="SUBSPACE"),
        rng=rng,
    )

    sim_gen_kw = list(
        prior_ens.load_parameters_numpy("SNAKE_OIL_PARAM", np.array([0])).flatten()
    )

    target_gen_kw = list(
        posterior_ens.load_parameters_numpy("SNAKE_OIL_PARAM", np.array([0])).flatten()
    )

    # Check that prior is not equal to posterior after updationg
    assert sim_gen_kw != target_gen_kw

    # Check that posterior is as expected
    assert target_gen_kw == pytest.approx(expected_gen_kw, abs=1e-5)


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

    resp = GenDataConfig(keys=["RESPONSE"])
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"gen_data": obs},
    )
    prior_storage = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    dataset = []
    for iens in range(prior_storage.ensemble_size):
        data = rng.uniform(0, 1)
        dataset.append(
            pl.DataFrame(
                {
                    "KEY_1": [data],
                    "realization": iens,
                }
            )
        )
        data = rng.uniform(0.8, 1, 3)
        prior_storage.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series(np.full(len(data), 0), dtype=pl.UInt16),
                    "index": pl.Series(range(len(data)), dtype=pl.UInt16),
                    "values": data,
                }
            ),
            iens,
        )
    prior_storage.save_parameters(dataset=pl.concat(dataset, how="vertical"))

    posterior_storage = storage.create_ensemble(
        prior_storage.experiment_id,
        ensemble_size=prior_storage.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_storage,
    )

    with expectation:
        result_snapshot = smoother_update(
            prior_storage,
            posterior_storage,
            observations=["OBSERVATION"],
            parameters=["KEY_1"],
            update_settings=ObservationSettings(
                outlier_settings=OutlierSettings(alpha=alpha)
            ),
            es_settings=ESSettings(inversion="SUBSPACE"),
            rng=rng,
        )
        assert result_snapshot.alpha == alpha
        assert (
            result_snapshot.observations_and_responses["status"].to_list() == expected
        )


@pytest.mark.integration_test
def test_update_only_using_subset_observations(
    snake_oil_case_storage, snake_oil_storage, snapshot
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage

    experiment = snake_oil_storage.get_experiment_by_name("ensemble-experiment")
    prior_ens = experiment.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.runpath_config.num_realizations,
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
        ObservationSettings(),
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
            {
                "INIT_FILES": "param_%d.GRDECL",
                "FORWARD_INIT": "False",
            },
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
        prior_ensemble.save_parameters(fields[iens], param_group, iens)

    realization_list = list(range(ensemble_size))
    param_ensemble_array = prior_ensemble.load_parameters_numpy(
        param_group, realization_list
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

    ensemble.save_parameters_numpy(param_ensemble_array, param_group, realization_list)
    for iens in range(prior_ensemble.ensemble_size):
        ds = xr.open_dataset(
            ensemble._path / f"realization-{iens}" / f"{param_group}.nc", engine="scipy"
        )
        np.testing.assert_array_equal(ds["values"].values[0], fields[iens]["values"])


def _mock_preprocess_observations_and_responses(
    observations_and_responses,
    observation_settings,
    global_std_scaling,
    progress_callback,
    ensemble,
):
    """
    Runs through _preprocess_observations_and_responses with mocked values for
     _get_observations_and_responses
    """
    with patch(
        "ert.storage.LocalEnsemble.get_observations_and_responses"
    ) as mock_obs_n_responses:
        mock_obs_n_responses.return_value = observations_and_responses

        return _preprocess_observations_and_responses(
            ensemble=ensemble,
            outlier_settings=observation_settings.outlier_settings,
            auto_scale_observations=observation_settings.auto_scale_observations,
            global_std_scaling=global_std_scaling,
            iens_active_index=np.array(
                [int(c) for c in observations_and_responses.columns[5:]]
            ),
            selected_observations=observations_and_responses.select("observation_key"),
            progress_callback=progress_callback,
        )


def test_that_autoscaling_applies_to_scaled_errors(storage):
    with patch("ert.analysis.misfit_preprocessor.main") as misfit_main:
        misfit_main.return_value = (
            np.array([2, 3]),
            np.array([1, 1]),  # corresponds to num obs keys in autoscaling group
            np.array([1, 1]),
        )

        observations_and_responses = pl.DataFrame(
            {
                "response_key": ["RESPONSE", "RESPONSE", "RESPONSE", "RESPONSE"],
                "index": ["rs00", "rs0", "rs0", "rs1"],
                "observation_key": ["obs1_1", "obs1_2", "obs2", "obs2"],
                "observations": pl.Series([2, 4, 3, 3], dtype=pl.Float32),
                "std": pl.Series([1, 2, 1, 1], dtype=pl.Float32),
                "1": pl.Series([1, 4, 7, 8], dtype=pl.Float32),
                "2": pl.Series([2, 5, 8, 11], dtype=pl.Float32),
                "3": pl.Series([3, 6, 9, 12], dtype=pl.Float32),
            }
        )

        outlier_settings = OutlierSettings(alpha=1, std_cutoff=0.05)
        global_std_scaling = 1

        def progress_callback(_):
            return None

        experiment = storage.create_experiment(name="dummyexp")
        ensemble = experiment.create_ensemble(name="dummy", ensemble_size=10)

        scaled_errors_with_autoscale = (
            _mock_preprocess_observations_and_responses(
                observations_and_responses,
                observation_settings=ObservationSettings(
                    outlier_settings=outlier_settings,
                    auto_scale_observations=[["obs1*"]],
                ),
                global_std_scaling=global_std_scaling,
                progress_callback=progress_callback,
                ensemble=ensemble,
            )
            .filter(pl.col("status") == ObservationStatus.ACTIVE)[
                _OutlierColumns.scaled_std
            ]
            .to_list()
        )

        scaled_errors_without_autoscale = (
            _mock_preprocess_observations_and_responses(
                observations_and_responses,
                observation_settings=ObservationSettings(
                    outlier_settings=outlier_settings, auto_scale_observations=[]
                ),
                global_std_scaling=global_std_scaling,
                progress_callback=progress_callback,
                ensemble=ensemble,
            )
            .filter(pl.col("status") == ObservationStatus.ACTIVE)[
                _OutlierColumns.scaled_std
            ]
            .to_list()
        )

        assert scaled_errors_with_autoscale == [2, 6]
        assert scaled_errors_without_autoscale == [1, 2]


@pytest.mark.parametrize(
    "nan_responses,overspread_responses,collapsed_responses",
    [
        pytest.param(set(), set(), set(), id="all ok"),
        pytest.param({0}, set(), set(), id="one nan response"),
        pytest.param(set(), {0}, set(), id="one overspread response"),
        pytest.param(set(), set(), {0}, id="one collapsed response"),
        pytest.param(set(range(10)), set(), set(), id="all nan responses"),
        pytest.param(set(), set(range(10)), set(), id="all overspread responses"),
        pytest.param(set(), set(), set(range(10)), id="all collapsed responses"),
        pytest.param({0}, {1}, {2}, id="all collapsed responses"),
        pytest.param({0, 2}, {1, 4}, {3, 8}, id="Mixed failures with some ok"),
        pytest.param({0, 3, 6, 9}, {1, 4, 7}, {2, 5, 8}, id="All mixed failures"),
    ],
)
def test_compute_observation_statuses(
    nan_responses, overspread_responses, collapsed_responses, caplog
):
    alpha = 0.1
    global_std_scaling = 1
    std_cutoff = 0.05

    num_reals = 10
    num_observations = 10

    rng = np.random.default_rng(42)

    responses_per_real = np.zeros((num_observations, num_reals), dtype=np.float32)
    observations = np.array(range(num_reals))
    observation_keys = [f"obs_{i}" for i in range(num_observations)]
    observation_errors = np.array([1] * num_reals)

    for obs_index in nan_responses:
        responses_per_real[obs_index, :] = np.nan

    for obs_index in overspread_responses:
        for real in range(num_reals):
            # Make the responses deviate ALOT from the observation
            # (Approximating to JUST ABOVE the cutoff would require some more logic)
            # Also ensure mean deviates from obs to avoid collapse
            responses_per_real[obs_index, real] += 3333 if real % 2 == 0 else -6666

    for obs_index in collapsed_responses:
        responses = (
            rng.standard_normal(num_reals) * (std_cutoff - 1e-6)
            + observations[obs_index]
        )
        responses_per_real[obs_index, :] = responses

    active_observations = (
        set(range(num_observations))
        - nan_responses
        - overspread_responses
        - collapsed_responses
    )
    for obs_index in active_observations:
        for real in range(num_reals):
            responses_per_real[obs_index, real] = observations[obs_index] + 1.5 * (
                -std_cutoff if real % 2 == 0 else std_cutoff
            )

    df = pl.DataFrame(
        {
            "observation_key": observation_keys,
            "observations": pl.Series(observations, dtype=pl.Float32),
            "std": pl.Series(observation_errors, dtype=pl.Float32),
            **{
                str(i): pl.Series(responses_per_real[:, i], dtype=pl.Float32)
                for i in range(num_reals)
            },
        }
    )

    df_with_statuses = _compute_observation_statuses(
        df,
        global_std_scaling=global_std_scaling,
        outlier_settings=OutlierSettings(alpha=alpha, std_cutoff=std_cutoff),
        active_realizations=[str(i) for i in range(num_reals)],
    )

    expected_statuses = np.array(
        [ObservationStatus.ACTIVE] * len(observation_keys), dtype="U20"
    )
    expected_statuses[list(nan_responses)] = str(ObservationStatus.MISSING_RESPONSE)
    expected_statuses[list(overspread_responses)] = str(ObservationStatus.OUTLIER)
    expected_statuses[list(collapsed_responses)] = str(ObservationStatus.STD_CUTOFF)

    assert expected_statuses.tolist() == df_with_statuses["status"].to_list()


def test_that_autoscaling_ignores_typos_in_observation_names(storage, caplog):
    observations_and_responses = pl.DataFrame(
        {
            "response_key": ["RESPONSE", "RESPONSE", "RESPONSE", "RESPONSE"],
            "index": ["rs00", "rs0", "rs0", "rs1"],
            "observation_key": ["obs1_1", "obs1_2", "obs2", "obs2"],
            "observations": pl.Series([2, 4, 3, 3], dtype=pl.Float32),
            "std": pl.Series([1, 2, 1, 1], dtype=pl.Float32),
            "1": pl.Series([1, 4, 7, 8], dtype=pl.Float32),
        }
    )

    experiment = storage.create_experiment(name="dummyexp")
    ensemble = experiment.create_ensemble(name="dummy", ensemble_size=10)
    _mock_preprocess_observations_and_responses(
        observations_and_responses,
        observation_settings=ObservationSettings(
            outlier_settings=OutlierSettings(alpha=1, std_cutoff=0.05),
            auto_scale_observations=[["OOOPS1*"]],
        ),
        global_std_scaling=1,
        progress_callback=lambda _: None,
        ensemble=ensemble,
    )
    logged_messages = str(caplog.messages)  # NB: The code also prints to the terminal
    assert "Could not auto-scale the observations" in logged_messages
    assert "OOPS" in logged_messages
    assert "obs1_1" in logged_messages


def test_that_deactivated_observations_are_logged(storage, caplog):
    observations_and_responses = pl.DataFrame(
        {
            "response_key": ["RESPONSE", "RESPONSE", "RESPONSE", "RESPONSE"],
            "index": ["rs00", "rs0", "rs0", "rs1"],
            "observation_key": ["obs1_1", "obs1_2", "obs2", "obs3"],
            "observations": pl.Series([2, 4, 3, 3], dtype=pl.Float32),
            "std": pl.Series([1, 2, 1, 1], dtype=pl.Float32),
            "1": pl.Series([1, 4, 7, 8], dtype=pl.Float32),
        }
    )

    experiment = storage.create_experiment(name="dummyexp")
    ensemble = experiment.create_ensemble(name="dummy", ensemble_size=10)
    _mock_preprocess_observations_and_responses(
        observations_and_responses,
        observation_settings=ObservationSettings(
            outlier_settings=OutlierSettings(alpha=1, std_cutoff=11111),
            auto_scale_observations=None,
        ),
        global_std_scaling=1,
        progress_callback=lambda _: None,
        ensemble=ensemble,
    )
    assert (
        "Deactivating observations: ['obs1_1', 'obs1_2', 'obs2', 'obs3']"
        in caplog.messages
    )


def test_that_activate_observations_are_not_logged_as_deactivated(storage, caplog):
    observations_and_responses = pl.DataFrame(
        {
            "response_key": ["RESPONSE", "RESPONSE", "RESPONSE", "RESPONSE"],
            "index": ["rs00", "rs0", "rs0", "rs1"],
            "observation_key": ["obs1_1", "obs1_2", "obs2", "obs3"],
            "observations": pl.Series([2, 4, 3, 3], dtype=pl.Float32),
            "std": pl.Series([1, 2, 1, 1], dtype=pl.Float32),
            "1": pl.Series([1, 4, 7, 8], dtype=pl.Float32),
            "3": pl.Series([1.1, 4.2, 7.5, 8.1], dtype=pl.Float32),
            "4": pl.Series([1.4, 4.3, 7.2, 7.4], dtype=pl.Float32),
            "5": pl.Series([1.4, 3.9, 6.8, 7.2], dtype=pl.Float32),
            "6": pl.Series([1.2, 4.1, 7.4, 9.1], dtype=pl.Float32),
            "8": pl.Series([1.9, 4.9, 7.1, 8.3], dtype=pl.Float32),
            "9": pl.Series([0.8, 3.5, 6.6, 7.9], dtype=pl.Float32),
        }
    )

    experiment = storage.create_experiment(name="dummyexp")
    ensemble = experiment.create_ensemble(name="dummy", ensemble_size=10)
    _mock_preprocess_observations_and_responses(
        observations_and_responses,
        observation_settings=ObservationSettings(
            outlier_settings=OutlierSettings(alpha=100, std_cutoff=0.000001),
            auto_scale_observations=None,
        ),
        global_std_scaling=1,
        progress_callback=lambda _: None,
        ensemble=ensemble,
    )
    assert not any("Deactivating observations" in m for m in caplog.messages)


def test_gen_data_obs_data_mismatch(storage, uniform_parameter):
    resp = GenDataConfig(keys=["RESPONSE"])
    gen_data_obs = pl.DataFrame(
        {
            "observation_key": "OBSERVATION",
            "response_key": ["RESPONSE"],
            "report_step": pl.Series([0], dtype=pl.UInt16),
            "index": pl.Series([1000], dtype=pl.UInt16),
            "observations": pl.Series([1.0], dtype=pl.Float32),
            "std": pl.Series([0.1], dtype=pl.Float32),
        }
    )

    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"gen_data": gen_data_obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    dataset = []
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        dataset.append(
            pl.DataFrame(
                {
                    "KEY_1": [data],
                    "realization": iens,
                }
            )
        )

        data = rng.uniform(0.8, 1, 3)
        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series([0] * len(data), dtype=pl.UInt16),
                    "index": pl.Series(range(len(data)), dtype=pl.UInt16),
                    "values": pl.Series(data, dtype=pl.Float32),
                }
            ),
            iens,
        )

    prior.save_parameters(dataset=pl.concat(dataset, how="vertical"))
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
            ["KEY_1"],
            ObservationSettings(),
            ESSettings(),
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_missing(storage, uniform_parameter, obs):
    resp = GenDataConfig(keys=["RESPONSE"])
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"gen_data": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    dataset = []
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        dataset.append(
            pl.DataFrame(
                {
                    "KEY_1": [data],
                    "realization": iens,
                }
            )
        )
        data = rng.uniform(0.8, 1, 2)  # Importantly, shorter than obs
        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series([0] * len(data), dtype=pl.UInt16),
                    "index": pl.Series(range(len(data)), dtype=pl.UInt16),
                    "values": pl.Series(data, dtype=pl.Float32),
                }
            ),
            iens,
        )
    prior.save_parameters(dataset=pl.concat(dataset, how="vertical"))
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
        ["KEY_1"],
        ObservationSettings(),
        ESSettings(),
        progress_callback=events.append,
    )

    assert update_snapshot.observations_and_responses["status"].to_list() == [
        ObservationStatus.ACTIVE,
        ObservationStatus.ACTIVE,
        ObservationStatus.MISSING_RESPONSE,
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_update_subset_parameters(storage, uniform_parameter, obs):
    no_update_param = GenKwConfig(
        name="KEY_2",
        group="EXTRA_PARAMETER",
        update=False,
        distribution={"name": "uniform", "min": 0, "max": 1},
    )
    resp = GenDataConfig(keys=["RESPONSE"])
    experiment = storage.create_experiment(
        parameters=[uniform_parameter, no_update_param],
        responses=[resp],
        observations={"gen_data": obs},
    )
    prior = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    dataset_key_1 = []
    dataset_key_2 = []
    for iens in range(prior.ensemble_size):
        data = rng.uniform(0, 1)
        dataset_key_1.append(
            pl.DataFrame(
                {
                    "KEY_1": [data],
                    "realization": iens,
                }
            )
        )
        dataset_key_2.append(
            pl.DataFrame(
                {
                    "KEY_2": [data],
                    "realization": iens,
                }
            )
        )

        data = rng.uniform(0.8, 1, 10)
        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series([0] * len(data), dtype=pl.UInt16),
                    "index": pl.Series(range(len(data)), dtype=pl.UInt16),
                    "values": pl.Series(data, dtype=pl.Float32),
                }
            ),
            iens,
        )

    prior.save_parameters(dataset=pl.concat(dataset_key_1, how="vertical"))
    prior.save_parameters(dataset=pl.concat(dataset_key_2, how="vertical"))
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
        ["KEY_1"],
        ObservationSettings(),
        ESSettings(),
    )

    assert (
        prior.load_parameters("EXTRA_PARAMETER", 0).rows()
        == posterior_ens.load_parameters("EXTRA_PARAMETER", 0).rows()
    )
    assert (
        prior.load_parameters("PARAMETER", 0).rows()
        != posterior_ens.load_parameters("PARAMETER", 0).rows()
    )
