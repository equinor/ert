from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ert.config import ConfigWarning, ErtConfig
from ert.config.analysis_config import UpdateSettings
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.validation import ActiveRange

from .base_run_model import BaseRunModel
from .ensemble_experiment import EnsembleExperiment
from .ensemble_smoother import EnsembleSmoother
from .evaluate_ensemble import EvaluateEnsemble
from .iterated_ensemble_smoother import IteratedEnsembleSmoother
from .multiple_data_assimilation import MultipleDataAssimilation
from .run_arguments import (
    EnsembleExperimentRunArguments,
    ESMDARunArguments,
    ESRunArguments,
    EvaluateEnsembleRunArguments,
    SIESRunArguments,
    SingleTestRunArguments,
)
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
        SingleTestRunArguments(
            random_seed=config.random_seed,
            ensemble_name=args.current_ensemble,
            minimum_required_realizations=1,
            ensemble_size=1,
            experiment_name=experiment_name,
            active_realizations=[True],
        ),
        config,
        storage,
        status_queue,
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
        ConfigWarning.ert_context_warn(
            f"Due to active_realizations {active_realizations_count} is lower than "
            f"MIN_REALIZATIONS {min_realizations_count}, MIN_REALIZATIONS has been "
            f"set to match active_realizations.",
        )
    experiment_name = args.experiment_name
    assert experiment_name is not None

    return EnsembleExperiment(
        EnsembleExperimentRunArguments(
            random_seed=config.random_seed,
            active_realizations=active_realizations.tolist(),
            ensemble_name=args.current_ensemble,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            experiment_name=experiment_name,
        ),
        config,
        storage,
        config.queue_config,
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
        ConfigWarning.ert_context_warn(
            "Adjusted MIN_REALIZATIONS to the current number of active realizations "
            f"({active_realizations_count}) as it is lower than the MIN_REALIZATIONS "
            f"({min_realizations_count}) that was specified in the config file."
        )

    return EvaluateEnsemble(
        EvaluateEnsembleRunArguments(
            random_seed=config.random_seed,
            active_realizations=active_realizations.tolist(),
            ensemble_id=args.ensemble_id,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
        ),
        config,
        storage,
        config.queue_config,
        status_queue=status_queue,
    )


def _setup_ensemble_smoother(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
    status_queue: SimpleQueue[StatusEvents],
) -> EnsembleSmoother:
    return EnsembleSmoother(
        ESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            target_ensemble=args.target_ensemble,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            experiment_name=getattr(args, "experiment_name", ""),
        ),
        config,
        storage,
        config.queue_config,
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

    return MultipleDataAssimilation(
        ESMDARunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            target_ensemble=_iterative_ensemble_format(config, args),
            weights=args.weights,
            restart_run=restart_run,
            prior_ensemble_id=prior_ensemble,
            starting_iteration=storage.get_ensemble(prior_ensemble).iteration + 1
            if restart_run
            else 0,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            experiment_name=args.experiment_name,
        ),
        config,
        storage,
        config.queue_config,
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
    return IteratedEnsembleSmoother(
        SIESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            target_ensemble=_iterative_ensemble_format(config, args),
            number_of_iterations=_num_iterations(config, args),
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            num_retries_per_iter=config.analysis_config.num_retries_per_iter,
            experiment_name=experiment_name,
        ),
        config,
        storage,
        config.queue_config,
        config.analysis_config.ies_module,
        update_settings=update_settings,
        status_queue=status_queue,
    )


def _realizations(args: Namespace, ensemble_size: int) -> npt.NDArray[np.bool_]:
    if args.realizations is None:
        return np.ones(ensemble_size, dtype=bool)
    return np.array(
        ActiveRange(rangestring=args.realizations, length=ensemble_size).mask
    )


def _iterative_ensemble_format(config: ErtConfig, args: Namespace) -> str:
    """
    When a RunModel runs multiple iterations, an ensemble format will be used.
    E.g. when starting from the ensemble 'ensemble', subsequent runs can be named
    'ensemble_0', 'ensemble_1', 'ensemble_2', etc.

    This format can be set from the commandline via the `target_ensemble` option,
    and via the config file via the `ITER_CASE` keyword. If none of these are
    set we use the name of the current ensemble and add `_%d` to it.
    """
    return (
        args.target_ensemble
        or config.analysis_config.ensemble_format
        or f"{getattr(args, 'current_ensemble', None) or 'default'}_%d"
    )


def _num_iterations(config: ErtConfig, args: Namespace) -> int:
    if args.num_iterations is not None:
        config.analysis_config.set_num_iterations(int(args.num_iterations))
    return config.analysis_config.num_iterations
