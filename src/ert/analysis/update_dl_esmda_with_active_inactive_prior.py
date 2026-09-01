from __future__ import annotations

import functools
import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Self, TextIO, TypeVar

import iterative_ensemble_smoother as ies
import numpy as np
import polars as pl
import psutil
import scipy
from iterative_ensemble_smoother.experimental import AdaptiveESMDA, DistanceESMDA

from ert.config import (
    ESSettings,
    GenKwConfig,
    ObservationSettings,
)

from ._update_commons import (
    ErtAnalysisError,
    _copy_unupdated_parameters,
    _OutlierColumns,
    _preprocess_observations_and_responses,
    noop_progress_callback,
)
from .event import (
    AnalysisCompleteEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    DataSection,
)
from .snapshots import (
    ObservationStatus,
    SmootherSnapshot,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


def calc_max_number_of_layers_per_batch_for_distance_localization(
    nx: int,
    ny: int,
    nz: int,
    num_obs: int,
    nreal: int,
    bytes_per_float: int = 8,  # float64 as default here
) -> int:
    """Calculate number of layers from a 3D field parameter that can be updated
    within available memory. Distance-based localization requires two large matrices
    the Kalman gain matrix K and the localization scaling matrix RHO, both have size
    equal to number of field parameter values times number of observations.
    Therefore, a batching algorithm is used where only a subset of parameters
    is used when calculating the Schur product of RHO and K matrix in the update
    algorithm. This function calculates number of batches and
    number of grid layers of field parameter values that can fit
    into the available memory for one batch accounting for a safety margin.

    The available memory is checked using the `psutil` library, which provides
    information about system memory usage.
    From `psutil` documentation:
    - available:
        the memory that can be given instantly to processes without the
        system going into swap.
        This is calculated by summing different memory values depending
        on the platform and it is supposed to be used to monitor actual
        memory usage in a cross platform fashion.

    Args:
        nx: grid size in I-direction (local x-axis direction)
        ny: grid size in J-direction (local y-axis direction)
        nz: grid size in K-direction (number of layers)
        num_obs: Number of observations
        nreal: Number of realizations
        bytes_per_float: Is 4 or 8

    Returns:
        Max number of layers that can be updated in one batch to
        avoid memory problems.

    """

    memory_safety_factor = 0.8
    num_params = nx * ny * nz
    num_param_per_layer = nx * ny

    # Rough estimate of necessary number of float variables
    sum_floats = 0
    sum_floats += num_params * num_obs  # K matrix before Schur product
    sum_floats += num_params * num_obs  # RHO matrix
    sum_floats += num_params * num_obs  # K matrix after Schur product
    sum_floats += int(num_params * nreal * 2.5)  # X_prior, X_prior_batch, M_delta
    sum_floats += int(num_params * nreal * 1.5)  # X_post and X_post_batch
    sum_floats += num_obs * nreal * 2  # D matrix and internal matrices
    sum_floats += num_obs * nreal * 2  # Y matrix and internal matrices

    # Check available memory
    available_memory_in_bytes = psutil.virtual_memory().available * memory_safety_factor

    # Required memory
    total_required_memory_per_field_param = sum_floats * bytes_per_float

    # Minimum number of batches
    min_number_of_batches = int(
        np.ceil(total_required_memory_per_field_param / available_memory_in_bytes)
    )

    max_nlayer_per_batch = int(nz / min_number_of_batches)

    if max_nlayer_per_batch == 0:
        # Batch size cannot be less than 1 layer
        memory_one_batch = num_param_per_layer * bytes_per_float
        raise MemoryError(
            "The required memory to update one grid layer or one 2D surface is "
            "larger than available memory.\n"
            "Cannot split the update into batch size less than one complete "
            "grid layer for 3D field or one surface for 2D fields."
            f"Required memory for one batch is about: {memory_one_batch / 10**9} GB\n"
            f"Available memory is about: {available_memory_in_bytes / 10**9} GB"
        )

    log_msg = (
        "Calculate batch size for updating of field parameter:\n"
        f" Number of parameters in field param: {num_params}\n"
        f" Required number of floats to update one field parameter: {sum_floats}\n"
        " Available memory per field param update: "
        f"{available_memory_in_bytes / 10**9} GB\n"
        " Required memory total to update a field parameter: "
        f"{total_required_memory_per_field_param / 10**9} GB\n"
        f" Number of layers in one batch: {max_nlayer_per_batch}"
    )
    logger.info(log_msg)
    return max_nlayer_per_batch


def update_3D_field_with_distance_esmda_with_active(
    distance_based_esmda_smoother: DistanceESMDA,
    field_param_name: str,
    X_prior: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    rho_2D: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    nz: int,
    min_real: int = 10,
    X_active_input: npt.NDArray[np.bool] | None = None,
    reshape_to_3D_per_realization: bool = False,
    min_nbatch: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool] | None]:
    """
    Calculate posterior update with distance-based ESMDA for one 3D parameter
    The RHO for one layer of the 3D field parameter is input.
    This is copied to all other layers of RHO in each batch of grid parameter
    layers since only lateral distance is used when calculating distances.
    Result is posterior parameter matrices of field parameters for one field.
    NOTE: Only field parameter values with a minimum number of realizations
    will be updated. Field parameters with less number of realizations than minimum will
    not be updated. If X_active is not defined, all field parameters will be updated
    if ensemble variance for the field parameters is > 0.

    Args:
        distance_based_esmda_smooter: Object of DistanceESMDA class initialized for use
        field_param_name: Name of 3D parameter
        X_prior: Matrix with prior realizations of all field parameters,
                 shape=(nparameters, nrealizations)
        Y: Matrix with response values for each observations for each realization,
                 shape=(nobservations, nrealizations)
        rho_2D: RHO matrix elements for one 3D grid layer with size (nx, ny),
                 shape=(nx,ny,nobservations)
        nx, ny, nz: Dimensions of the 3D grid filled with a 3D field parameter.
        min_real: Minimum number of realizations required for a field parameter
                 to be updated.
        X_active_input: Matrix with False or True. X_active_input has
                  shape (nparameters, nrealizations)
                  X_active_input[parameter_number, real_number] = True for
                  field parameter realizations to be used and updated
                  and X_active_input[parameter_number, real_number] = False
                  for field parameter realizations not to be used and updated.
                  This input can be used e.g. for geological zones where
                  some grid cells are active for some realizations but not
                  for all realizations.
        reshape_to_3D_per_realization: Is set to True if output field is reshaped to 3D
                  per realization.
        min_nbatch: Minimum number of batches the field parameter is split into.
                  Default is 1. Usually number of batches will be calculated based
                  on available memory and the size of the field parameters,
                  number of observations and realizations.
                  The actual number of batches will be
                  max(min_nbatch, min_number_of_batches_required)
    Results:
        X_post: Posterior ensemble of field parameters,
          shape=(nx*ny*nz, nrealizations) if not reshaped and
          shape=(nx,ny,nz, nrealizations) if reshaped to 3D per realization
        X_active: Matrix with active parameters. This will be created if it is not
          defined as input. If it is defined as input, it should not be changed
          and return the same as output.
          shape=(nx*ny*nz, nrealizations) if not reshaped and
          shape=(nx,ny,nz, nrealizations) if reshaped to 3D per realization
    """
    nparam_per_layer = nx * ny
    nparam = nparam_per_layer * nz
    nreal = X_prior.shape[1]
    assert X_prior.shape[0] == nparam, (
        f"Mismatch between X_prior dimension {X_prior.shape[0]} and nparam {nparam}"
    )

    X_prior_3D = X_prior.reshape(nx, ny, nz, nreal)
    # No update if no observations or responses
    if Y is None or Y.shape[0] == 0:
        # No update of the field parameters
        # Check if it necessary to make a copy or can we only return X_prior?
        if reshape_to_3D_per_realization:
            return X_prior_3D.copy(), None
        return X_prior.copy(), None

    nobs = Y.shape[0]
    assert Y.shape[1] == nreal, (
        f"Mismatch between X_prior dimension {Y.shape[1]} and nreal {nreal}"
    )

    log_msg = f"Calculate Distance-based ESMDA update for {field_param_name} "
    log_msg += f"with {nparam} parameters"
    logger.info(log_msg)

    # Posterior ensemble will not be modified for field parameters that are
    # not updatable. Initialize it to be equal to prior ensemble and update it
    # later for updatable field parameter
    X_post_3D = X_prior_3D.copy()

    # Check memory constraints and calculate how many grid layers of
    # field parameters is possible to update on one batch
    max_nlayers_per_batch = (
        calc_max_number_of_layers_per_batch_for_distance_localization(
            nx, ny, nz, nobs, nreal, bytes_per_float=8
        )
    )  # Use float64

    if X_active_input is None:
        # No X_active matrix is supplied, but there can be field parameters
        # with 0 ensemble variance. They will be defined as inactive.
        #
        # This option ( not to define X_active as input) will require
        # from the users that the field parameters are defined for all
        # realizations for all field parameters. But if there are
        # grid cells in ertbox not corresponding to any physical grid cell
        # in the geomodel zone, the field parameters for those grid cells
        # in ertbox grid should be assigned a sensible value.
        # If it is known that some field parameters will belong to ertbox
        # grid cells that never is used in any realization in the geomodel grid,
        # the user can assign a constant value, e.g. 0 to all realizations of these
        # parameters. They will then have 0 variance and not be used in the update.
        # They will never be used to update the geogrid field parameters either
        # since they never is used as active values in the geomodel grid.

        # Define a matrix of same size and shape as the X_prior
        # The values are True if the parameter is updatable and
        # False if not (due to 0 standard deviation)
        X_active, updatable_field_param, _ = define_active_matrix_for_initial_ensemble(
            X_prior, None, min_real=min_real
        )

        # X_prior is NOT modified here.
        # X_active is later used to select only updatable field parameters

    else:
        # In this case the assumption is as follows:
        #  - The input X_active mark which parameters come from active and
        #    inactive geomodel grid cells for all realizations.
        #    The algorithm below will also calculate a value to assign
        #    to field parameters corresponding to geomodel grid cells that
        #    are not active in all realizations, but for some realizations.
        #    The assigned value will be the ensemble average of the field value
        #    based upon the realizations where the geomodel grid cell is active.
        #    This ensures that the update algorithm will get an ensemble average
        #    that is equal to the average over the active realizations
        #    even though the average is taken over all realizations.
        #    This ensures that no bias in the ensemble average will happen
        #    due to untypical/unphysical values in field parameter values
        #    corresponding to inactive geomodel grid cells in some or all
        #    realizations. In this way it simplifies the forward model
        #    to simulate prior ensemble. The user does not have to think
        #    about extrapolating or assigning sensible values for field
        #    parameter values corresponding to inactive grid cells in
        #    the realizations of the geomodel grid.
        #  - The forward model simulating field parameter values must, however,
        #    supply ERT with an active/inactive parameter per realization
        #    such that it can be possible to filter out inactive values.
        #
        # The code below will use the information about which field parameters
        # is active or not for each realization.
        #
        # Example: if X_active[param_number, real_number] is True for
        # a set of real_number which is less than min_real realizations,
        # the values X_prior[param_number,real_number]  is set to the average over
        # the active realizations for the param_number where
        # X_active[param_number,real_number] is False.
        # This ensure that the average over all realization is equal to the average
        # over the active realizations:
        # X_prior[param_number,:].mean(axis=1) =
        # X_prior[param_number,active_realizations].mean(axis=1)

        # The X_active_input matrix should here be up to date
        # but need updatable_field_param and number_of_updatable
        X_active_original = X_active_input.copy()
        X_active, updatable_field_param, _ = define_active_matrix_for_initial_ensemble(
            X_prior,
            X_active_input,
            min_real=min_real,
        )
        # This check ensure that input X_active is consistent with the X_prior
        # such that 0 variance parameters are inactivated and that number of
        # realizations with active parameters are at least the minimum specified.

        compare_X_active = X_active == X_active_original
        if not np.all(compare_X_active):
            raise ValueError("Expecting X_active input not be modified.")

        # If there are field parameters with some inactive realizations and
        # and some active realizations, the values for the inactive realizations
        # are modified to get a value equal to the average of the active realizations
        X_prior = adjust_inactive_field_values_to_match_average_of_active_field_values(
            X_prior, X_active
        )

    # Remove layers with only inactive parameters
    updatable_field_param_3D = updatable_field_param.reshape(nx, ny, nz)

    updatable_layers = np.sum(updatable_field_param_3D, axis=(0, 1)) > 0
    number_of_updatable_layers = np.sum(updatable_layers)

    assert number_of_updatable_layers <= nz

    # Calculate number of batches based on memory constraint and field size,
    # number of observations and realizations
    nlayer_per_batch = min(max_nlayers_per_batch, number_of_updatable_layers)
    nbatch = int(number_of_updatable_layers / nlayer_per_batch)

    # If number of batches is specified, used number of batches may be increased.
    nbatch = max(min_nbatch, nbatch)
    nlayer_per_batch = int(number_of_updatable_layers / nbatch)

    X_prior_3D = X_prior.reshape(nx, ny, nz, nreal)
    X_active_3D = X_active.reshape(nx, ny, nz, nreal)

    # Number of parameters in a batch including both active and inactive parameters
    nparam_in_batch = nparam_per_layer * nlayer_per_batch
    # The remaining number of grid layers of field parameters
    nlayer_last_batch = number_of_updatable_layers - nbatch * nlayer_per_batch

    if nlayer_last_batch > 0:
        log_msg = f"Number of batches to update {field_param_name} is {nbatch + 1}"
        logger.info(log_msg)
    else:
        log_msg = f"Number of batches to update {field_param_name} is {nbatch}"
        logger.info(log_msg)

    # First update all batches with a full set of grid layers with field parameters
    # The algorithm will skip grid layers with field parameters where all parameters
    # are not updatable. No need to spend time and memory on those.
    start_layer_number = 0
    for batch_number in range(nbatch):
        # Choose nlayer_per_batch layers with field parameters among updatable layers
        selected_layers, end_layer_number = get_updatable_layers(
            updatable_layers, start_layer_number, nlayer_per_batch, nz
        )

        log_msg = (
            f"Batch number: {batch_number}\n"
            f"start layer : {start_layer_number}\n"
            f"end layer   : {end_layer_number - 1}"
        )
        logger.info(log_msg)

        # Only whole layers of inactive parameters are removed in this selection
        X_batch = X_prior_3D[:, :, selected_layers, :].reshape((nparam_in_batch, nreal))
        X_batch_active = X_active_3D[:, :, selected_layers, :].reshape(
            (nparam_in_batch, nreal)
        )

        # Define indices for parameters in reshaped array that
        # have active realizations.
        active_params = np.sum(X_batch_active, axis=1) > 0

        # Keep only field parameters with some or all active realizations
        X_batch_filtered = X_batch[active_params, :]

        # The rho matrix for the batch of field parameters
        rho_3D_batch = np.zeros((nx, ny, nlayer_per_batch, nobs), dtype=np.float64)

        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers. Shape of rho_2D is (nx, ny, nobs)
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_batch, nobs))

        # Keep only rho values corresponding to the parameters that
        # is updatable and have active realizations.
        rho_batch_filtered = rho_batch[active_params, :]
        assert rho_batch_filtered.shape[0] == X_batch_filtered.shape[0]

        log_msg = f"Assimilate batch number {batch_number}"
        logger.info(log_msg)
        X_post_batch = X_batch
        print(f"Assimilate batch {batch_number}")
        X_post_batch_filtered = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch_filtered, Y=Y, rho_batch=rho_batch_filtered
        )
        # Update the active params, keep the inactive params unchanged
        X_post_batch[active_params, :] = X_post_batch_filtered[:, :]
        X_post_3D[:, :, selected_layers, :] = X_post_batch.reshape(
            nx, ny, nlayer_per_batch, nreal
        )
        start_layer_number = end_layer_number

    if nlayer_last_batch > 0:
        batch_number = nbatch
        selected_layers, end_layer_number = get_updatable_layers(
            updatable_layers, start_layer_number, nlayer_last_batch, nz
        )

        assert len(selected_layers) == nlayer_last_batch

        nparam_in_last_batch = nparam_per_layer * nlayer_last_batch

        log_msg = f"Batch number: {batch_number}\n"
        log_msg += f"start layer : {start_layer_number}\n"
        log_msg += f"end layer   : {end_layer_number - 1}"
        logger.info(log_msg)

        X_batch = X_prior_3D[:, :, selected_layers, :].reshape(
            (nparam_in_last_batch, nreal)
        )
        X_batch_active = X_active_3D[:, :, selected_layers, :].reshape(
            (nparam_in_last_batch, nreal)
        )
        # Define indices for parameters in reshaped array that
        # have active realizations
        active_params = np.sum(X_batch_active, axis=1) > 0

        # Keep only field parameters with some or all active realizations
        X_batch_filtered = X_batch[active_params, :]

        # The rho matrix for the batch of field parameters
        rho_3D_batch = np.zeros((nx, ny, nlayer_last_batch, nobs), dtype=np.float64)

        # Copy rho calculated from one layer of 3D parameter into all layers for
        # current batch of layers. Shape of rho_2D is (nx, ny, nobs)
        rho_3D_batch[:, :, :, :] = rho_2D[:, :, np.newaxis, :]
        rho_batch = rho_3D_batch.reshape((nparam_in_last_batch, nobs))

        # Keep only rho values corresponding to the parameters that
        # is updatable and have active realizations.
        rho_batch_filtered = rho_batch[active_params, :]
        assert rho_batch_filtered.shape[0] == X_batch_filtered.shape[0]

        log_msg = f"Assimilate batch number {batch_number}"
        logger.info(log_msg)
        X_post_batch = X_batch
        print(f"Assimilate batch {batch_number}")
        X_post_batch_filtered = distance_based_esmda_smoother.assimilate_batch(
            X_batch=X_batch_filtered, Y=Y, rho_batch=rho_batch_filtered
        )
        # Update the active params, keep the inactive params unchanged
        X_post_batch[active_params, :] = X_post_batch_filtered[:, :]
        X_post_3D[:, :, selected_layers, :] = X_post_batch.reshape(
            nx, ny, nlayer_last_batch, nreal
        )
    if reshape_to_3D_per_realization:
        return X_post_3D, X_active_3D
    else:
        return X_post_3D.reshape(nparam, nreal), X_active_3D.reshape(nparam, nreal)


def get_updatable_layers(
    updatable_layers: npt.NDArray[np.bool],
    start_layer_number: int,
    nlayers_per_batch: int,
    nz: int,
) -> tuple[list[int], int]:
    count = 0
    layers = []
    end_layer_number = start_layer_number
    for n in range(start_layer_number, nz):
        if updatable_layers[n]:
            if count < nlayers_per_batch:
                layers.append(n)
                count += 1
            else:
                end_layer_number = n
                break
    return layers, end_layer_number


def define_active_matrix_for_initial_ensemble(
    X_matrix: npt.NDArray[np.float64],
    X_active: npt.NDArray[np.bool] | None = None,
    min_real: int = 10,
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool], int]:
    """Create the X_active matrix or update it if it already exists.
    Uses two criteria to deactivate field parameters (mark them as not updatable)
    Criteria 1: If ensemble standard devation is 0 (all realizations are equal),
    the field parameter can not be updated and must be deactivated.
    Criteria 2: If a field parameter has both active and inactive realizations
    set the field parameter inactive if number of realizations is below a minimum.

    Args:
        X_matrix -  Ensemble of field parameters. Shape = (nparam, nreal)
        X_active -  Same size as X_matrix and contains True for field parameters
                    and realizations where the field parameter is active
                    and False if not active.
        min_real -  Minimum number of active realizations for the field parameter to
                    be defined as updatable.
    Returns:
        X_active matrix - created if input X_active is None
                    or updated if X_active input exists.
        updatable_field_params - Bool array with True for active field parameter
                    and False for inactive.
        number_of updatable_field_params  - Number of active parameters

    """
    assert X_matrix is not None
    nparam = X_matrix.shape[0]
    nreal = X_matrix.shape[1]

    if X_active is None:
        # Initialized to active for all field parameters for all realizations
        X_active = np.full((nparam, nreal), True, dtype=np.bool)

    print(f"{X_active.shape=}")
    # Check critera 1: Has some field parameters 0 ensemble std?
    X_std = X_matrix.std(axis=1)
    non_updatable = X_std == 0

    if np.sum(non_updatable) > 0:
        # Some field parameters have 0 variance
        X_active[non_updatable, :] = False

    # Check criteria 2: Has sufficient number of realizations?
    params_with_too_few_realizations = X_active.sum(axis=1) < min_real
    X_active[params_with_too_few_realizations, :] = False
    updatable_field_params = np.sum(X_active, axis=1) >= min_real
    number_of_updatable_params = np.sum(updatable_field_params)
    return X_active, updatable_field_params, number_of_updatable_params


def adjust_inactive_field_values_to_match_average_of_active_field_values(
    X_matrix: npt.NDArray[np.float64],
    X_active: npt.NDArray[np.bool],
) -> npt.NDArray[np.float64]:
    """This function will modify the ensemble of field parameters in X_matrix
    and return a modified version. The purpose is to ensure that field parameter
    realizations that are inactive don't contribute to the ensemble average
    which is taken over all field parameter realizations (both active and inactive)
    in the DistanceESMDA assimilate algorithm. This is done by calculating
    the ensemble average over the active realizations and
    assign the average value to the field parameter realizations defined as inactive.


    The steps in the algorithm is then:
    - Calculate average over the ensemble of the active (used) field parameter
      values for each individual field parameter.
    - For each field parameter having some active realizations, assign the
      ensemble average to the realizations that are defined as inactive.
    - For field parameters that are inactive for all realizations, do nothing.


    Args:
        X_matrix - Input ensemble of field parameters, shape = (nparam, nreal)
        X_active - Input matrix defining which field parameter is active
                   or inactive for each realization, shape = (nparam, nreal)
    Returns:
        Modified X_matrix

    """
    assert X_matrix is not None
    assert X_active is not None

    inactive = ~X_active
    # Safely compute the ensemble mean (row-wise mean) for active entries only
    active_real_per_field_param = np.sum(X_active, axis=1, keepdims=True)

    # Avoid division by zero by setting mean to 0
    # if there are no active realizations for a field parameter
    active_real_per_field_param[active_real_per_field_param == 0] = 1
    X_mean = np.zeros(X_matrix.shape, dtype=np.float64)
    X_mean[:, :] = (
        np.sum(X_matrix * X_active, axis=1, keepdims=True) / active_real_per_field_param
    )

    # Replace field parameters with some inactive realizations
    # with the average value over the active realizations.
    # Field parameters that are inactive are set to 0
    X_matrix[inactive] = X_mean[inactive]

    # Return the modified matrix
    return X_matrix
