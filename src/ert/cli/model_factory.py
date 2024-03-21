from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.config import ConfigWarning, ErtConfig
from ert.config.analysis_config import UpdateSettings
from ert.run_models import (
    BaseRunModel,
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
    ESMDARunArguments,
    ESRunArguments,
    EvaluateEnsembleRunArguments,
    SIESRunArguments,
    SingleTestRunArguments,
)
from ert.validation import ActiveRange

if TYPE_CHECKING:

    import numpy.typing as npt

    from ert.namespace import Namespace
    from ert.storage import Storage


def create_model(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
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
        return _setup_single_test_run(config, storage, args)
    elif args.mode == ENSEMBLE_EXPERIMENT_MODE:
        return _setup_ensemble_experiment(config, storage, args)
    elif args.mode == EVALUATE_ENSEMBLE_MODE:
        return _setup_evaluate_ensemble(config, storage, args)
    elif args.mode == ENSEMBLE_SMOOTHER_MODE:
        return _setup_ensemble_smoother(config, storage, args, update_settings)
    elif args.mode == ES_MDA_MODE:
        return _setup_multiple_data_assimilation(config, storage, args, update_settings)
    elif args.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE:
        return _setup_iterative_ensemble_smoother(
            config, storage, args, update_settings
        )

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")


def _setup_single_test_run(
    config: ErtConfig, storage: Storage, args: Namespace
) -> SingleTestRun:
    return SingleTestRun(
        SingleTestRunArguments(
            random_seed=config.random_seed,
            current_ensemble=args.current_ensemble,
            minimum_required_realizations=1,
            ensemble_size=1,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=args.experiment_name,
        ),
        config,
        storage,
    )


def _setup_ensemble_experiment(
    config: ErtConfig, storage: Storage, args: Namespace
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
            current_ensemble=args.current_ensemble,
            iter_num=int(
                getattr(args, "iter_num", 0)
            ),  # Safely access iter_num with 0 as a default
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=experiment_name,
        ),
        config,
        storage,
        config.queue_config,
    )


def _setup_evaluate_ensemble(
    config: ErtConfig, storage: Storage, args: Namespace
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
            current_ensemble=args.ensemble_name,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=None,
        ),
        config,
        storage,
        config.queue_config,
    )


def _setup_ensemble_smoother(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
) -> EnsembleSmoother:
    return EnsembleSmoother(
        ESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            current_ensemble=args.current_ensemble,
            target_ensemble=args.target_ensemble,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=getattr(args, "experiment_name", ""),
        ),
        config,
        storage,
        config.queue_config,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
    )


def _determine_restart_info(args: Namespace) -> Tuple[bool, str]:
    """Handles differences in configuration between CLI and GUI.

    Returns
    -------
    A tuple containing the restart_run flag and the ensemble
    to run from.
    """
    if hasattr(args, "restart_ensemble"):
        # When running from CLI
        restart_run = args.restart_ensemble is not None
        prior_ensemble = args.restart_ensemble
    else:
        # When running from GUI
        restart_run = args.restart_run
        prior_ensemble = args.prior_ensemble
    return restart_run, prior_ensemble


def _setup_multiple_data_assimilation(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
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
            prior_ensemble=prior_ensemble,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=args.experiment_name,
        ),
        config,
        storage,
        config.queue_config,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
    )


def _setup_iterative_ensemble_smoother(
    config: ErtConfig,
    storage: Storage,
    args: Namespace,
    update_settings: UpdateSettings,
) -> IteratedEnsembleSmoother:
    return IteratedEnsembleSmoother(
        SIESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            current_ensemble=args.current_ensemble,
            target_ensemble=_iterative_ensemble_format(config, args),
            num_iterations=_num_iterations(config, args),
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            num_retries_per_iter=config.analysis_config.num_retries_per_iter,
            stop_long_running=config.analysis_config.stop_long_running,
            experiment_name=args.experiment_name,
        ),
        config,
        storage,
        config.queue_config,
        config.analysis_config.ies_module,
        update_settings=update_settings,
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
