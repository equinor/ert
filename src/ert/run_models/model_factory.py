from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.config.analysis_config import UpdateSettings
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    MANUAL_UPDATE_MODE,
    TEST_RUN_MODE,
)
from ert.validation import ActiveRange

from .base_run_model import BaseRunModel
from .ensemble_experiment import EnsembleExperiment
from .ensemble_smoother import EnsembleSmoother
from .evaluate_ensemble import EvaluateEnsemble
from .iterated_ensemble_smoother import IteratedEnsembleSmoother
from .manual_update import ManualUpdate
from .multiple_data_assimilation import MultipleDataAssimilation
from .single_test_run import SingleTestRun

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.namespace import Namespace
    from ert.run_models.base_run_model import StatusEvents
    from ert.storage import Storage


def create_model(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    status_queue: SimpleQueue[StatusEvents],
) -> BaseRunModel:
    logger = logging.getLogger(__name__)
    logger.info(
        "Initiating experiment",
        extra={
            "mode": args.mode,
            "ensemble_size": config.model_config.num_realizations,
        },
    )
    update_settings = config.analysis_config.observation_settings

    if args.mode == TEST_RUN_MODE:
        return _setup_single_test_run(config, storage, args, status_queue)
    elif args.mode == ENSEMBLE_EXPERIMENT_MODE:
        return _setup_ensemble_experiment(config, storage, args, status_queue)
    elif args.mode == EVALUATE_ENSEMBLE_MODE:
        return _setup_evaluate_ensemble(config, storage, args, status_queue)
    elif args.mode == ENSEMBLE_SMOOTHER_MODE:
        return _setup_ensemble_smoother(
            config, storage, args, update_settings, status_queue
        )
    elif args.mode == ES_MDA_MODE:
        return _setup_multiple_data_assimilation(
            config, storage, args, update_settings, status_queue
        )
    elif args.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE:
        return _setup_iterative_ensemble_smoother(
            config, storage, args, update_settings, status_queue
        )
    elif args.mode == MANUAL_UPDATE_MODE:
        return _setup_manual_update(
            config, storage, args, update_settings, status_queue
        )

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")


def _setup_single_test_run(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    status_queue: SimpleQueue[StatusEvents],
) -> SingleTestRun:
    experiment_name = (
        "single-test-run" if args.experiment_name is None else args.experiment_name
    )

    return SingleTestRun(
        random_seed=config.random_seed,
        ensemble_name=args.current_ensemble,
        experiment_name=experiment_name,
        config=config,
        storage=storage,
        status_queue=status_queue,
    )


def _setup_ensemble_experiment(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    status_queue: SimpleQueue[StatusEvents],
) -> EnsembleExperiment:
    min_realizations_count = config.analysis_config.minimum_required_realizations
    active_realizations = _realizations(args, config.model_config.num_realizations)
    active_realizations_count = int(np.sum(active_realizations))
    if active_realizations_count < min_realizations_count:
        config.analysis_config.minimum_required_realizations = active_realizations_count
        ConfigWarning.warn(
            f"Due to active_realizations {active_realizations_count} is lower than "
            f"MIN_REALIZATIONS {min_realizations_count}, MIN_REALIZATIONS has been "
            f"set to match active_realizations.",
        )
    experiment_name = args.experiment_name
    assert experiment_name is not None

    return EnsembleExperiment(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        ensemble_name=args.current_ensemble,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        experiment_name=experiment_name,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        status_queue=status_queue,
    )


def _setup_evaluate_ensemble(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    status_queue: SimpleQueue[StatusEvents],
) -> EvaluateEnsemble:
    min_realizations_count = config.analysis_config.minimum_required_realizations
    active_realizations = _realizations(args, config.model_config.num_realizations)
    active_realizations_count = int(np.sum(active_realizations))
    if active_realizations_count < min_realizations_count:
        config.analysis_config.minimum_required_realizations = active_realizations_count
        ConfigWarning.warn(
            "Adjusted MIN_REALIZATIONS to the current number of active realizations "
            f"({active_realizations_count}) as it is lower than the MIN_REALIZATIONS "
            f"({min_realizations_count}) that was specified in the config file."
        )

    return EvaluateEnsemble(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        ensemble_id=args.ensemble_id,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        status_queue=status_queue,
    )


def _validate_num_realizations(
    args: Namespace, config: ErtConfig
) -> npt.NDArray[np.bool_]:
    active_realizations = _realizations(args, config.model_config.num_realizations)
    if int(np.sum(active_realizations)) <= 1:
        raise ConfigValidationError(
            "Number of active realizations must be at least 2 for an update step"
        )
    return active_realizations


def _setup_manual_update(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
    status_queue: SimpleQueue[StatusEvents],
) -> ManualUpdate:
    min_realizations_count = config.analysis_config.minimum_required_realizations
    active_realizations = _realizations(args, config.model_config.num_realizations)
    active_realizations_count = int(np.sum(active_realizations))
    if active_realizations_count < min_realizations_count:
        config.analysis_config.minimum_required_realizations = active_realizations_count
        ConfigWarning.warn(
            "Adjusted MIN_REALIZATIONS to the current number of active realizations "
            f"({active_realizations_count}) as it is lower than the MIN_REALIZATIONS "
            f"({min_realizations_count}) that was specified in the config file."
        )

    return ManualUpdate(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        ensemble_id=args.ensemble_id,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        target_ensemble=args.target_ensemble,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
        status_queue=status_queue,
    )


def _setup_ensemble_smoother(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
    status_queue: SimpleQueue[StatusEvents],
) -> EnsembleSmoother:
    active_realizations = _validate_num_realizations(args, config)
    return EnsembleSmoother(
        target_ensemble=args.target_ensemble,
        experiment_name=getattr(args, "experiment_name", ""),
        active_realizations=active_realizations.tolist(),
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        random_seed=config.random_seed,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
        status_queue=status_queue,
    )


def _determine_restart_info(args: Namespace) -> Tuple[bool, str]:
    """Handles differences in configuration between CLI and GUI.

    Returns
    -------
    A tuple containing the restart_run flag and the ensemble
    to run from.
    """
    if hasattr(args, "restart_ensemble_id"):
        # When running from CLI
        restart_run = args.restart_ensemble_id is not None
        prior_ensemble = args.restart_ensemble_id
    else:
        # When running from GUI
        restart_run = args.restart_run
        prior_ensemble = args.prior_ensemble_id
    return restart_run, prior_ensemble


def _setup_multiple_data_assimilation(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
    status_queue: SimpleQueue[StatusEvents],
) -> MultipleDataAssimilation:
    restart_run, prior_ensemble = _determine_restart_info(args)
    active_realizations = _validate_num_realizations(args, config)
    return MultipleDataAssimilation(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        target_ensemble=_iterative_ensemble_format(args),
        weights=args.weights,
        restart_run=restart_run,
        prior_ensemble_id=prior_ensemble,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        experiment_name=args.experiment_name,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
        status_queue=status_queue,
    )


def _setup_iterative_ensemble_smoother(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
    status_queue: SimpleQueue[StatusEvents],
) -> IteratedEnsembleSmoother:
    experiment_name = "ies" if args.experiment_name is None else args.experiment_name
    active_realizations = _validate_num_realizations(args, config)
    return IteratedEnsembleSmoother(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        target_ensemble=_iterative_ensemble_format(args),
        number_of_iterations=int(args.num_iterations)
        if args.num_iterations is not None
        else 4,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        num_retries_per_iter=4,
        experiment_name=experiment_name,
        config=config,
        storage=storage,
        queue_config=config.queue_config,
        analysis_config=config.analysis_config.ies_module,
        update_settings=update_settings,
        status_queue=status_queue,
    )


def _realizations(args: Namespace, ensemble_size: int) -> npt.NDArray[np.bool_]:
    if args.realizations is None:
        return np.ones(ensemble_size, dtype=bool)
    return np.array(
        ActiveRange(rangestring=args.realizations, length=ensemble_size).mask
    )


def _iterative_ensemble_format(args: Namespace) -> str:
    """
    When a RunModel runs multiple iterations, an ensemble format will be used.
    E.g. when starting from the ensemble 'ensemble', subsequent runs can be named
    'ensemble_0', 'ensemble_1', 'ensemble_2', etc.

    This format can be set from the commandline via the `target_ensemble` option.
    or we use the current ensemble and add `_%d` to it.
    """
    return (
        args.target_ensemble
        or f"{getattr(args, 'current_ensemble', None) or 'default'}_%d"
    )
