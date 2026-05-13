import iterative_ensemble_smoother as ies
import numpy as np
import pytest

from ert.analysis._update_strategies._distance import DistanceLocalizationUpdate
from ert.analysis._update_strategies._protocol import (
    ObservationContext,
    ObservationLocations,
)
from ert.config import Field, SurfaceConfig
from ert.field_utils import AxisOrientation, ErtboxParameters, FieldFileFormat
from ert.storage import open_storage


def _noop_callback(_event: object) -> None:
    pass


def _field_config(nx: int, ny: int, nz: int) -> Field:
    return Field(
        name="TEST_FIELD",
        ertbox_params=ErtboxParameters(
            nx=nx,
            ny=ny,
            nz=nz,
            xinc=1.0,
            yinc=1.0,
            origin=(0.0, 0.0),
            rotation_angle=0.0,
            axis_orientation=AxisOrientation.LEFT_HANDED,
        ),
        file_format=FieldFileFormat.ROFF,
        forward_init_file="init_%d.roff",
        forward_init=False,
        update=True,
        output_file="output.roff",
        grid_file="dummy.grdecl",
    )


def _surface_config(ncol: int, nrow: int) -> SurfaceConfig:
    return SurfaceConfig(
        name="TEST_SURFACE",
        forward_init=False,
        update=True,
        ncol=ncol,
        nrow=nrow,
        xori=0.0,
        yori=0.0,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        yflip=1,
        forward_init_file="init_%d.irap",
        output_file="output.irap",
        base_surface_path="base.irap",
    )


def test_that_distance_localization_updates_all_z_layers_at_observation_xy(
    tmp_path,
):
    """
    Places a single observation in the corner cell (0, 0) with a short correlation
    range so only that xy cell gets rho ≈ 1. With correct z-expansion,
    parameters at (0, 0, z) for ALL z should be updated while distant xy
    cells are untouched.
    """
    nx, ny, nz = 4, 4, 3
    n_params = nx * ny * nz
    n_real = 50
    xinc, yinc = 1.0, 1.0

    field_config = _field_config(nx, ny, nz)

    rng = np.random.default_rng(42)
    raw_params = rng.standard_normal((n_params, n_real))

    # Round-trip through real storage to get the actual parameter ordering
    realizations = np.arange(n_real, dtype=np.int_)
    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment_id = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [field_config.model_dump(mode="json")]
            }
        )
        ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=n_real
        )
        ensemble.save_parameters_numpy(raw_params, "TEST_FIELD", realizations)
        param_ensemble = ensemble.load_parameters_numpy("TEST_FIELD", realizations)

    # Reshape flat arrays to (nx, ny, nz, n_real) using C-order to index by grid cell
    prior_3d = param_ensemble.reshape(nx, ny, nz, n_real)

    # Observation near cell (x=0, y=0) — centre of first cell
    obs_x = np.array([0.5 * xinc])
    obs_y = np.array([0.5 * yinc])
    main_range = np.array([1.0])

    # Responses correlated with parameters at (x=0, y=0) — average over all z-layers.
    responses = prior_3d[0, 0, :, :].mean(axis=0, keepdims=True).astype(np.float64)

    obs_values = np.array([5.0])
    obs_errors = np.array([0.1])

    obs_loc = ObservationLocations(
        xpos=obs_x,
        ypos=obs_y,
        main_range=main_range,
        location_mask=np.ones(responses.shape[0], dtype=bool),
    )

    rng = np.random.default_rng(42)
    obs_context = ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
        observation_perturbations=rng.standard_normal(
            size=(responses.shape[0], responses.shape[1])
        ).astype(np.float64)
        * obs_errors[:, np.newaxis],
        observation_locations=obs_loc,
    )

    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=Field,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble.copy(),
        param_config=field_config,
        non_zero_variance_mask=np.ones(n_params, dtype=bool),
    )

    posterior_3d = posterior.reshape(nx, ny, nz, n_real)

    # All z-layers at the observation cell (x=0, y=0) must be updated.
    for iz in range(nz):
        assert not np.allclose(posterior_3d[0, 0, iz, :], prior_3d[0, 0, iz, :]), (
            f"z-layer {iz} at (x=0, y=0) was NOT updated -- "
            f"rho z-expansion does not match parameter storage order."
        )

    # Distant xy cell should be untouched (localization weight near zero)
    assert np.allclose(
        posterior_3d[nx // 2, ny // 2, :, :], prior_3d[nx // 2, ny // 2, :, :]
    ), "Distant xy cell was updated despite near-zero localization weight."

    # Variance should be reduced at the observation cell
    for iz in range(nz):
        prior_var = np.var(prior_3d[0, 0, iz, :])
        posterior_var = np.var(posterior_3d[0, 0, iz, :])
        assert posterior_var < prior_var, (
            f"Variance not reduced at z-layer {iz} of observation cell."
        )


@pytest.mark.parametrize(
    ("param_type", "param_config", "n_params"),
    [
        (Field, _field_config(3, 3, 1), 9),
        (SurfaceConfig, _surface_config(3, 3), 9),
    ],
)
def test_that_unlocated_observations_are_assimilated_globally_for_distance_update(
    param_type,
    param_config,
    n_params,
):
    n_real = 30
    rng = np.random.default_rng(123)
    param_ensemble = rng.standard_normal((n_params, n_real))

    responses = np.vstack(
        [
            param_ensemble[0],
            param_ensemble[-1],
        ]
    )
    obs_values = np.array([3.0, -2.0])
    obs_errors = np.array([0.2, 0.3])
    observation_perturbations = (
        rng.standard_normal(size=responses.shape) * obs_errors[:, np.newaxis]
    )

    obs_context = ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
        observation_perturbations=observation_perturbations,
        observation_locations=None,
    )

    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=param_type,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble.copy(),
        param_config=param_config,
        non_zero_variance_mask=np.ones(n_params, dtype=bool),
    )

    smoother = ies.ESMDA(
        covariance=obs_errors**2,
        observations=obs_values,
        alpha=1,
        seed=None,
    )
    smoother.prepare_assimilation(
        Y=responses,
        truncation=0.99,
        overwrite=False,
        observation_perturbations=observation_perturbations,
    )
    expected = smoother.assimilate_batch(X=param_ensemble.copy(), overwrite=True)

    np.testing.assert_allclose(posterior, expected)


@pytest.mark.parametrize(
    ("param_type", "param_config", "n_params"),
    [
        (Field, _field_config(3, 3, 1), 9),
        (SurfaceConfig, _surface_config(3, 3), 9),
    ],
)
def test_that_unlocated_distance_update_respects_non_zero_variance_mask(
    param_type,
    param_config,
    n_params,
) -> None:
    n_real = 30
    rng = np.random.default_rng(124)
    param_ensemble = rng.standard_normal((n_params, n_real))
    prior = param_ensemble.copy()

    responses = np.vstack([param_ensemble[0], param_ensemble[-1]])
    obs_errors = np.array([0.2, 0.3])
    obs_context = ObservationContext(
        responses=responses,
        observation_values=np.array([3.0, -2.0]),
        observation_errors=obs_errors,
        observation_perturbations=rng.standard_normal(size=responses.shape)
        * obs_errors[:, np.newaxis],
        observation_locations=None,
    )

    non_zero_variance_mask = np.ones(n_params, dtype=bool)
    non_zero_variance_mask[0] = False

    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=param_type,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble,
        param_config=param_config,
        non_zero_variance_mask=non_zero_variance_mask,
    )

    np.testing.assert_allclose(posterior[0, :], prior[0, :])
    assert not np.allclose(posterior[1:, :], prior[1:, :])


def test_that_distance_localization_matrix_uses_float32() -> None:
    rng = np.random.default_rng(125)
    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=Field,
        progress_callback=_noop_callback,
    )
    obs_context = ObservationContext(
        responses=rng.standard_normal((2, 10)),
        observation_values=np.ones(2),
        observation_errors=np.ones(2),
        observation_perturbations=np.zeros((2, 10)),
        observation_locations=ObservationLocations(
            xpos=np.array([0.0]),
            ypos=np.array([0.0]),
            main_range=np.array([1.0]),
            location_mask=np.array([True, False]),
        ),
    )
    updater.prepare(obs_context)

    rho = updater._full_localization_matrix(
        3,
        np.array([[0.5], [0.25], [0.0]], dtype=np.float64),
    )

    assert rho.dtype == np.float32
    np.testing.assert_allclose(rho[:, 0], np.array([0.5, 0.25, 0.0]))
    np.testing.assert_allclose(rho[:, 1], np.ones(3))


def test_that_mixed_unlocated_observations_update_distant_field_parameters(
    monkeypatch,
):
    """
    Verifies that unlocated observations (location_mask=False) update all
    correlated parameters globally, even when mixed with located observations
    that use distance localization. The distant cell (3, 3) is outside the
    correlation range of the located observation at (0.5, 0.5), but should
    still be updated because it correlates with the unlocated observation.
    """
    nx, ny, nz = 4, 4, 1
    n_params = nx * ny * nz
    n_real = 40
    rng = np.random.default_rng(321)
    param_ensemble = rng.standard_normal((n_params, n_real))

    prior_3d = param_ensemble.reshape(nx, ny, nz, n_real)
    distant_x, distant_y = 3, 3
    responses = np.vstack(
        [
            prior_3d[0, 0, 0, :],
            prior_3d[distant_x, distant_y, 0, :],
        ]
    )
    obs_values = np.array([2.0, -4.0])
    obs_errors = np.array([0.1, 0.1])
    observation_perturbations = (
        rng.standard_normal(size=responses.shape) * obs_errors[:, np.newaxis]
    )

    obs_context = ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
        observation_perturbations=observation_perturbations,
        observation_locations=ObservationLocations(
            xpos=np.array([0.5]),
            ypos=np.array([0.5]),
            main_range=np.array([1.0]),
            location_mask=np.array([True, False]),
        ),
    )

    forced_batch_size = 5
    monkeypatch.setattr(
        "ert.analysis._update_strategies._distance.calculate_localization_batch_size",
        lambda _num_params, _num_obs: forced_batch_size,
    )

    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=Field,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble.copy(),
        param_config=_field_config(nx, ny, nz),
        non_zero_variance_mask=np.ones(n_params, dtype=bool),
    )

    posterior_3d = posterior.reshape(nx, ny, nz, n_real)
    assert not np.allclose(
        posterior_3d[distant_x, distant_y, 0, :], prior_3d[distant_x, distant_y, 0, :]
    )


def test_that_mixed_unlocated_observations_update_distant_surface_parameters(
    monkeypatch,
):
    ncol, nrow = 4, 4
    n_params = ncol * nrow
    n_real = 40
    rng = np.random.default_rng(321)
    param_ensemble = rng.standard_normal((n_params, n_real))

    prior_2d = param_ensemble.reshape(ncol, nrow, n_real)
    distant_x, distant_y = 3, 3
    responses = np.vstack(
        [
            prior_2d[0, 0, :],
            prior_2d[distant_x, distant_y, :],
        ]
    )
    obs_values = np.array([2.0, -4.0])
    obs_errors = np.array([0.1, 0.1])
    observation_perturbations = (
        rng.standard_normal(size=responses.shape) * obs_errors[:, np.newaxis]
    )

    obs_context = ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
        observation_perturbations=observation_perturbations,
        observation_locations=ObservationLocations(
            xpos=np.array([0.5]),
            ypos=np.array([0.5]),
            main_range=np.array([1.0]),
            location_mask=np.array([True, False]),
        ),
    )

    forced_batch_size = 5
    monkeypatch.setattr(
        "ert.analysis._update_strategies._distance.calculate_localization_batch_size",
        lambda _num_params, _num_obs: forced_batch_size,
    )

    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=SurfaceConfig,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble.copy(),
        param_config=_surface_config(ncol, nrow),
        non_zero_variance_mask=np.ones(n_params, dtype=bool),
    )

    posterior_2d = posterior.reshape(ncol, nrow, n_real)
    assert not np.allclose(
        posterior_2d[distant_x, distant_y, :], prior_2d[distant_x, distant_y, :]
    )


def test_that_distance_localization_batch_size_does_not_change_result(
    monkeypatch,
) -> None:
    nx, ny, nz = 4, 4, 1
    n_params = nx * ny * nz
    n_real = 30
    rng = np.random.default_rng(17)
    param_ensemble = rng.standard_normal((n_params, n_real))
    field_config = _field_config(nx, ny, nz)
    responses = np.vstack(
        [
            param_ensemble.reshape(nx, ny, nz, n_real)[0, 0, 0, :],
            param_ensemble.reshape(nx, ny, nz, n_real)[2, 2, 0, :],
        ]
    )
    obs_context = ObservationContext(
        responses=responses,
        observation_values=np.array([2.0, -1.0]),
        observation_errors=np.array([0.1, 0.2]),
        observation_perturbations=rng.standard_normal((2, n_real))
        * np.array([[0.1], [0.2]]),
        observation_locations=ObservationLocations(
            xpos=np.array([0.5, 2.5]),
            ypos=np.array([0.5, 2.5]),
            main_range=np.array([2.0, 2.0]),
            location_mask=np.ones(2, dtype=bool),
        ),
    )

    def update_with_batch_size(batch_size: int) -> np.ndarray:
        monkeypatch.setattr(
            "ert.analysis._update_strategies._distance.calculate_localization_batch_size",
            lambda _num_params, _num_obs: batch_size,
        )
        updater = DistanceLocalizationUpdate(
            enkf_truncation=0.99,
            param_type=Field,
            progress_callback=_noop_callback,
        )
        updater.prepare(obs_context)
        return updater.update(
            param_ensemble=param_ensemble.copy(),
            param_config=field_config,
            non_zero_variance_mask=np.ones(n_params, dtype=bool),
        )

    np.testing.assert_allclose(
        update_with_batch_size(1),
        update_with_batch_size(4),
    )


def test_that_distance_localization_respects_non_zero_variance_mask(
    monkeypatch,
) -> None:
    nx, ny, nz = 3, 3, 2
    n_params = nx * ny * nz
    n_real = 30
    rng = np.random.default_rng(11)
    param_ensemble = rng.standard_normal((n_params, n_real))
    prior = param_ensemble.copy()
    field_config = _field_config(nx, ny, nz)
    responses = param_ensemble[:1, :].copy()
    obs_context = ObservationContext(
        responses=responses,
        observation_values=np.array([3.0]),
        observation_errors=np.array([0.1]),
        observation_perturbations=rng.standard_normal((1, n_real)) * 0.1,
        observation_locations=ObservationLocations(
            xpos=np.array([0.5]),
            ypos=np.array([0.5]),
            main_range=np.array([5.0]),
            location_mask=np.ones(1, dtype=bool),
        ),
    )
    non_zero_variance_mask = np.ones(n_params, dtype=bool)
    non_zero_variance_mask[0] = False

    forced_batch_size = 4
    monkeypatch.setattr(
        "ert.analysis._update_strategies._distance.calculate_localization_batch_size",
        lambda _num_params, _num_obs: forced_batch_size,
    )
    updater = DistanceLocalizationUpdate(
        enkf_truncation=0.99,
        param_type=Field,
        progress_callback=_noop_callback,
    )
    updater.prepare(obs_context)

    posterior = updater.update(
        param_ensemble=param_ensemble,
        param_config=field_config,
        non_zero_variance_mask=non_zero_variance_mask,
    )

    np.testing.assert_allclose(posterior[0, :], prior[0, :])
    assert not np.allclose(posterior[1:, :], prior[1:, :])
