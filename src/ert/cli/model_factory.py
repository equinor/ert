from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
)
from ert.config import ConfigWarning, HookRuntime
from ert.enkf_main import EnKFMain
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
    ert: EnKFMain,
    storage: StorageAccessor,
    args: Namespace,
    experiment_id: UUID,
) -> BaseRunModel:
    logger = logging.getLogger(__name__)
    logger.info(
        "Initiating experiment",
        extra={
            "mode": args.mode,
            "ensemble_size": ert.ert_config.model_config.num_realizations,
        },
    )

    if args.mode == TEST_RUN_MODE:
        return _setup_single_test_run(ert, storage, args, experiment_id)
    elif args.mode == ENSEMBLE_EXPERIMENT_MODE:
        return _setup_ensemble_experiment(ert, storage, args, experiment_id)
    elif args.mode == ENSEMBLE_SMOOTHER_MODE:
        return _setup_ensemble_smoother(ert, storage, args, experiment_id)
    elif args.mode == ES_MDA_MODE:
        return _setup_multiple_data_assimilation(ert, storage, args, experiment_id)
    elif args.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE:
        return _setup_iterative_ensemble_smoother(ert, storage, args, experiment_id)

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")


def _setup_single_test_run(
    ert: EnKFMain, storage: StorageAccessor, args: Namespace, experiment_id: UUID
) -> SingleTestRun:
    return SingleTestRun(
        SingleTestRunArguments(
            random_seed=ert.ert_config.random_seed,
            current_case=args.current_case,
            minimum_required_realizations=1,
            ensemble_size=1,
        ),
        ert,
        storage,
        experiment_id,
    )


def _setup_ensemble_experiment(
    ert: EnKFMain, storage: StorageAccessor, args: Namespace, experiment_id: UUID
) -> EnsembleExperiment:
    config = ert.ert_config
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
            random_seed=ert.ert_config.random_seed,
            active_realizations=active_realizations.tolist(),
            current_case=args.current_case,
            iter_num=int(args.iter_num),
            minimum_required_realizations=config.analysis_config.minimum_required_realizations,
            ensemble_size=config.model_config.num_realizations,
        ),
        ert,
        storage,
        config.queue_config,
        experiment_id,
    )


def _setup_ensemble_smoother(
    ert: EnKFMain, storage: StorageAccessor, args: Namespace, experiment_id: UUID
) -> EnsembleSmoother:
    return EnsembleSmoother(
        ESRunArguments(
            random_seed=ert.ert_config.random_seed,
            active_realizations=_realizations(
                args, ert.ert_config.model_config.num_realizations
            ).tolist(),
            current_case=args.current_case,
            target_case=args.target_case,
            minimum_required_realizations=ert.ert_config.analysis_config.minimum_required_realizations,
            ensemble_size=ert.ert_config.model_config.num_realizations,
            misfit_process=_misfit_preprocessor(
                ert.ert_config.hooked_workflows[HookRuntime.PRE_FIRST_UPDATE]
            ),
        ),
        ert,
        storage,
        ert.ert_config.queue_config,
        experiment_id,
    )


def _setup_multiple_data_assimilation(
    ert: EnKFMain, storage: StorageAccessor, args: Namespace, experiment_id: UUID
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
            random_seed=ert.ert_config.random_seed,
            active_realizations=_realizations(
                args, ert.ert_config.model_config.num_realizations
            ).tolist(),
            target_case=_target_case_name(ert, args, format_mode=True),
            weights=args.weights,
            restart_run=restart_run,
            prior_ensemble=prior_ensemble,
            minimum_required_realizations=ert.ert_config.analysis_config.minimum_required_realizations,
            ensemble_size=ert.ert_config.model_config.num_realizations,
            misfit_process=_misfit_preprocessor(
                ert.ert_config.hooked_workflows[HookRuntime.PRE_FIRST_UPDATE]
            ),
        ),
        ert,
        storage,
        ert.ert_config.queue_config,
        experiment_id,
        prior_ensemble,
    )


def _setup_iterative_ensemble_smoother(
    ert: EnKFMain, storage: StorageAccessor, args: Namespace, id_: UUID
) -> IteratedEnsembleSmoother:
    return IteratedEnsembleSmoother(
        SIESRunArguments(
            random_seed=ert.ert_config.random_seed,
            active_realizations=_realizations(
                args, ert.ert_config.model_config.num_realizations
            ).tolist(),
            current_case=args.current_case,
            target_case=_target_case_name(ert, args, format_mode=True),
            num_iterations=_num_iterations(ert, args),
            minimum_required_realizations=ert.ert_config.analysis_config.minimum_required_realizations,
            ensemble_size=ert.ert_config.model_config.num_realizations,
            num_retries_per_iter=ert.ert_config.analysis_config.num_retries_per_iter,
            misfit_process=_misfit_preprocessor(
                ert.ert_config.hooked_workflows[HookRuntime.PRE_FIRST_UPDATE]
            ),
        ),
        ert,
        storage,
        ert.ert_config.queue_config,
        id_,
    )


def _realizations(args: Namespace, ensemble_size: int) -> npt.NDArray[np.bool_]:
    if args.realizations is None:
        return np.ones(ensemble_size, dtype=bool)
    return np.array(
        ActiveRange(rangestring=args.realizations, length=ensemble_size).mask
    )


def _target_case_name(ert: EnKFMain, args: Namespace, format_mode: bool = False) -> str:
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        return f"{args.current_case}_smoother_update"

    analysis_config = ert.ert_config.analysis_config
    if analysis_config.case_format is not None:
        return analysis_config.case_format

    return f"{args.current_case}_%d"


def _num_iterations(ert: EnKFMain, args: Namespace) -> int:
    if args.num_iterations is not None:
        ert.ert_config.analysis_config.set_num_iterations(int(args.num_iterations))
    return ert.ert_config.analysis_config.num_iterations
