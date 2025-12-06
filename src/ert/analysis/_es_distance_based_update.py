import numpy as np
import numpy.typing as npt
from iterative_ensemble_smoother.experimental import DistanceESMDA


def update_3D_field_with_distance_esmda(
    distance_based_esmda_smoother: DistanceESMDA,
    field_param_name: str,
    X_prior: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    rho_2D: npt.NDArray[np.float64],
    nlayer_per_batch: int,
    nx: int,
    ny: int,
    nz: int,
) -> npt.NDArray[np.float64]:
    """
    Calculate posterior update with distance-based ESMDA for one 3D parameter
    The RHO for one layer of the 3D field parameter is input.
    This is copied to all other layers of RHO in each batch of grid parameter
    layers since only lateral distance is used when calculating distances.
    Result is prior and posterior parameter matrices of field parameters for one field.

    Args:
        distance_based_esmda_smooter: Object of DistanceESMDA class initialized for use
           field_param_name: Name of 3D parameter
        X_prior: Matrix with prior realizations of all field parameters,
           shape=(nparameters, nrealizations)
        Y: Matrix with response values for each observations for each realization,
           shape=(nobservations, nrealizations)
        rho_2D: RHO matrix elements for one 3D grid layer with size (nx, ny),
           shape=(nx,ny,nobservations)
        nlayer_per_batch: Number of 3D grid layers that are included in
           one batch of field parameters.
        nx, ny, nz: Dimensions of the 3D grid filled with a 3D field parameter.

    Results:
        X_post_3D: Posterior ensemble of field parameters,
          shape=(nx, ny, nz, nrealizations)
    """
    nparam_per_layer = nx * ny
    nobs = Y.shape[0]
    nreal = Y.shape[1]

    assert X_prior.shape[0] == nparam_per_layer * nz
    assert X_prior.shape[1] == nreal
    nparam = nparam_per_layer * nz
    print(f"    Calculate update for {field_param_name} with {nparam} parameters")

    X_prior_3D = X_prior.reshape(nx, ny, nz, nreal)
    X_post_3D = np.zeros((nx, ny, nz, nreal), dtype=np.float64)
    nlayer_per_batch = min(nlayer_per_batch, nz)
    nbatch = int(nz / nlayer_per_batch)

    nparam_in_batch = (
        nparam_per_layer * nlayer_per_batch
    )  # For full sized batch of layers

    nlayer_last_batch = nz - nbatch * nlayer_per_batch
    for batch_number in range(nbatch):
        start_layer_number = batch_number * nlayer_per_batch
        end_layer_number = start_layer_number + nlayer_per_batch

        print(
            f"Batch number: {batch_number} "
            f"start: {start_layer_number} "
            f"end: {end_layer_number - 1}"
        )

        X_batch = X_prior_3D[:, :, start_layer_number:end_layer_number, :].reshape(
            (nparam_in_batch, nreal)
        )

        print(f"Define rho for 3D field {field_param_name}")
        rho_3D_batch = np.zeros((nx, ny, nlayer_per_batch, nobs), dtype=np.float64)
        print(
            "Memory required for RHO and K matrix in this batch: "
            f"{2 * nx * ny * nlayer_per_batch * nobs * 64 / 10**9}GB"
        )
        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers.
        # Size of rho batch: (nx,ny,nlayer_per_batch,nobs)
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_batch, nobs))

        print(f"Assimilate batch number {batch_number}")
        X_post_batch = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch, Y=Y, rho_batch=rho_batch
        )
        X_post_3D[:, :, start_layer_number:end_layer_number, :] = X_post_batch.reshape(
            nx, ny, nlayer_per_batch, nreal
        )

    if nlayer_last_batch > 0:
        batch_number = nbatch
        start_layer_number = batch_number * nlayer_per_batch
        end_layer_number = start_layer_number + nlayer_last_batch
        nparam_in_last_batch = nparam_per_layer * nlayer_last_batch
        print(
            f"Batch number: {batch_number} "
            f"start: {start_layer_number} "
            f"end: {end_layer_number - 1}"
        )

        X_batch = X_prior_3D[:, :, start_layer_number:end_layer_number, :].reshape(
            (nparam_in_last_batch, nreal)
        )
        print(f"Define rho for 3D field {field_param_name}")

        rho_3D_batch = np.zeros((nx, ny, nlayer_last_batch, nobs), dtype=np.float64)
        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers
        # Size of rho batch: (nx,ny,nlayer_per_batch,nobs)
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_last_batch, nobs))

        print(f"Assimilate batch number {batch_number}")
        X_post_batch = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch, Y=Y, rho_batch=rho_batch
        )
        X_post_3D[:, :, start_layer_number:end_layer_number, :] = X_post_batch.reshape(
            nx, ny, nlayer_last_batch, nreal
        )

    return X_post_3D
