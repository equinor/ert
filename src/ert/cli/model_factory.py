from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.analysis._es_update import UpdateSettings
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.config import ConfigWarning, ErtConfig, HookRuntime
from ert.run_models import (
    BaseRunModel,
    EnsembleExperiment,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
)
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
    ESMDARunArguments,
    ESRunArguments,
    SIESRunArguments,
    SingleTestRunArguments,
)
from ert.validation import ActiveRange

if TYPE_CHECKING:
    from typing import List

    import numpy.typing as npt

    from ert.config import Workflow
    from ert.namespace import Namespace
    from ert.storage import StorageAccessor


def _misfit_preprocessor(workflows: List[Workflow]) -> bool:
    for workflow in workflows:
        for job, _ in workflow:
            if job.name == "MISFIT_PREPROCESSOR":
                return True
    return False


def create_model(
    config: ErtConfig,
    storage: StorageAccessor,
    args: Namespace,
    experiment_id: UUID,
) -> BaseRunModel:
    logger = logging.getLogger(__name__)
    logger.info(
        "Initiating experiment",
        extra={
            "mode": args.mode,
            "ensemble_size": config.model_config.num_realizations,
        },
    )
    ert_analysis_config = config.analysis_config
    update_settings = UpdateSettings(
        std_cutoff=ert_analysis_config.std_cutoff,
        alpha=ert_analysis_config.enkf_alpha,
        misfit_preprocess=_misfit_preprocessor(
            config.hooked_workflows[HookRuntime.PRE_FIRST_UPDATE]
        ),
        min_required_realizations=ert_analysis_config.minimum_required_realizations,
    )

    if args.mode == TEST_RUN_MODE:
        return _setup_single_test_run(config, storage, args, experiment_id)
    elif args.mode == ENSEMBLE_EXPERIMENT_MODE:
        return _setup_ensemble_experiment(config, storage, args, experiment_id)
    elif args.mode == ENSEMBLE_SMOOTHER_MODE:
        return _setup_ensemble_smoother(
            config, storage, args, experiment_id, update_settings
        )
    elif args.mode == ES_MDA_MODE:
        return _setup_multiple_data_assimilation(
            config, storage, args, experiment_id, update_settings
        )
    elif args.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE:
        return _setup_iterative_ensemble_smoother(
            config, storage, args, experiment_id, update_settings
        )

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")


def _setup_single_test_run(
    config: ErtConfig, storage: StorageAccessor, args: Namespace, experiment_id: UUID
) -> SingleTestRun:
    return SingleTestRun(
        SingleTestRunArguments(
            random_seed=config.random_seed,
            current_case=args.current_case,
            minimum_required_realizations=1,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
        ),
        config,
        storage,
        experiment_id,
    )


def _setup_ensemble_experiment(
    config: ErtConfig, storage: StorageAccessor, args: Namespace, experiment_id: UUID
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

    return EnsembleExperiment(
        EnsembleExperimentRunArguments(
            random_seed=config.random_seed,
            active_realizations=active_realizations.tolist(),
            current_case=args.current_case,
            iter_num=int(args.iter_num),
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
        ),
        config,
        storage,
        config.queue_config,
        experiment_id,
    )


def _setup_ensemble_smoother(
    config: ErtConfig,
    storage: StorageAccessor,
    args: Namespace,
    experiment_id: UUID,
    update_settings: UpdateSettings,
) -> EnsembleSmoother:
    return EnsembleSmoother(
        ESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            current_case=args.current_case,
            target_case=args.target_case,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
        ),
        config,
        storage,
        config.queue_config,
        experiment_id,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
    )


def _setup_multiple_data_assimilation(
    config: ErtConfig,
    storage: StorageAccessor,
    args: Namespace,
    experiment_id: UUID,
    update_settings: UpdateSettings,
) -> MultipleDataAssimilation:
    # Because the configuration of the CLI is different from the gui, we
    # have a different way to get the restart information.
    if hasattr(args, "restart_case"):
        restart_run = args.restart_case is not None
        prior_ensemble = args.restart_case
    else:
        restart_run = args.restart_run
        prior_ensemble = args.prior_ensemble
    return MultipleDataAssimilation(
        ESMDARunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            target_case=_target_case_name(config, args, format_mode=True),
            weights=args.weights,
            restart_run=restart_run,
            prior_ensemble=prior_ensemble,
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            stop_long_running=config.analysis_config.stop_long_running,
        ),
        config,
        storage,
        config.queue_config,
        experiment_id,
        prior_ensemble,
        es_settings=config.analysis_config.es_module,
        update_settings=update_settings,
    )


def _setup_iterative_ensemble_smoother(
    config: ErtConfig,
    storage: StorageAccessor,
    args: Namespace,
    id_: UUID,
    update_settings: UpdateSettings,
) -> IteratedEnsembleSmoother:
    return IteratedEnsembleSmoother(
        SIESRunArguments(
            random_seed=config.random_seed,
            active_realizations=_realizations(
                args, config.model_config.num_realizations
            ).tolist(),
            current_case=args.current_case,
            target_case=_target_case_name(config, args, format_mode=True),
            num_iterations=_num_iterations(config, args),
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
            num_retries_per_iter=config.analysis_config.num_retries_per_iter,
            stop_long_running=config.analysis_config.stop_long_running,
        ),
        config,
        storage,
        config.queue_config,
        id_,
        config.analysis_config.ies_module,
        update_settings=update_settings,
    )


def _realizations(args: Namespace, ensemble_size: int) -> npt.NDArray[np.bool_]:
    if args.realizations is None:
        return np.ones(ensemble_size, dtype=bool)
    return np.array(
        ActiveRange(rangestring=args.realizations, length=ensemble_size).mask
    )


def _target_case_name(
    config: ErtConfig, args: Namespace, format_mode: bool = False
) -> str:
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        return f"{args.current_case}_smoother_update"

    analysis_config = config.analysis_config
    if analysis_config.case_format is not None:
        return analysis_config.case_format

    if not hasattr(args, "current_case"):
        return "default_%d"

    return f"{args.current_case}_%d"


def _num_iterations(config: ErtConfig, args: Namespace) -> int:
    if args.num_iterations is not None:
        config.analysis_config.set_num_iterations(int(args.num_iterations))
    return config.analysis_config.num_iterations
