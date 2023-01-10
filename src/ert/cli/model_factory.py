from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List
from uuid import UUID

from ert._c_wrappers.config.active_range import ActiveRange
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert.shared.models.ensemble_experiment import EnsembleExperiment
from ert.shared.models.ensemble_smoother import EnsembleSmoother
from ert.shared.models.iterated_ensemble_smoother import IteratedEnsembleSmoother
from ert.shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert.shared.models.single_test_run import SingleTestRun

if TYPE_CHECKING:
    from ert.storage import StorageAccessor


def create_model(
    ert: EnKFMain,
    storage: StorageAccessor,
    args,
    experiment_id: UUID,
):
    logger = logging.getLogger(__name__)
    logger.info(
        "Initiating experiment",
        extra={
            "mode": args.mode,
            "ensemble_size": ert.getEnsembleSize(),
        },
    )

    if args.mode == "test_run":
        return _setup_single_test_run(ert, storage, args, experiment_id)
    elif args.mode == "ensemble_experiment":
        return _setup_ensemble_experiment(ert, storage, args, experiment_id)
    elif args.mode == "ensemble_smoother":
        return _setup_ensemble_smoother(ert, storage, args, experiment_id)
    elif args.mode == "es_mda":
        return _setup_multiple_data_assimilation(ert, storage, args, experiment_id)
    elif args.mode == "iterative_ensemble_smoother":
        return _setup_iterative_ensemble_smoother(ert, storage, args, experiment_id)

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")


def _setup_single_test_run(ert, storage, args, experiment_id):
    simulations_argument = {
        "active_realizations": [True],
        "current_case": args.current_case,
    }
    model = SingleTestRun(simulations_argument, ert, storage, experiment_id)
    return model


def _setup_ensemble_experiment(ert, storage, args, experiment_id):
    simulations_argument = {
        "active_realizations": _realizations(args, ert.getEnsembleSize()),
        "iter_num": int(args.iter_num),
        "current_case": args.current_case,
    }
    model = EnsembleExperiment(
        simulations_argument, ert, storage, ert.get_queue_config(), experiment_id
    )
    return model


def _setup_ensemble_smoother(ert: EnKFMain, storage, args, experiment_id):
    simulations_argument = {
        "active_realizations": _realizations(args, ert.getEnsembleSize()),
        "current_case": args.current_case,
        "target_case": _target_case_name(ert, args, format_mode=False),
        "analysis_module": "STD_ENKF",
    }
    model = EnsembleSmoother(
        simulations_argument,
        ert,
        storage,
        ert.get_queue_config(),
        experiment_id,
    )
    return model


def _setup_multiple_data_assimilation(ert, storage, args, experiment_id):
    # Because the configuration of the CLI is different from the gui, we
    # infer the restart status from the starting iteration.
    if not hasattr(args, "restart_run"):
        restart_run = int(args.start_iteration) != 0
        prior_ensemble = (
            None if not restart_run else storage.get_ensemble_by_name(args.current_case)
        )
    else:
        restart_run = args.restart_run
        prior_ensemble = args.prior_ensemble
    simulations_argument = {
        "active_realizations": _realizations(args, ert.getEnsembleSize()),
        "current_case": args.current_case,
        "target_case": _target_case_name(ert, args, format_mode=True),
        "analysis_module": "STD_ENKF",
        "weights": args.weights,
        "start_iteration": int(args.start_iteration),
        "num_iterations": len(args.weights),
        "restart_run": restart_run,
        "prior_ensemble": prior_ensemble,
    }
    model = MultipleDataAssimilation(
        simulations_argument,
        ert,
        storage,
        ert.get_queue_config(),
        experiment_id,
        prior_ensemble,
    )
    return model


def _setup_iterative_ensemble_smoother(ert, storage, args, id_):
    simulations_argument = {
        "active_realizations": _realizations(args, ert.getEnsembleSize()),
        "current_case": args.current_case,
        "target_case": _target_case_name(ert, args, format_mode=True),
        "analysis_module": "IES_ENKF",
        "num_iterations": _num_iterations(ert, args),
    }
    model = IteratedEnsembleSmoother(
        simulations_argument, ert, storage, ert.get_queue_config(), id_
    )
    return model


def _realizations(args, ensemble_size: int) -> List[bool]:
    if args.realizations is None:
        return [True] * ensemble_size
    return ActiveRange(rangestring=args.realizations, length=ensemble_size).mask


def _target_case_name(ert, args, format_mode=False) -> str:
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        return f"{args.current_case}_smoother_update"

    analysis_config = ert.analysisConfig()
    if analysis_config.case_format_is_set():
        return analysis_config.case_format

    return f"{args.current_case}_%d"


def _num_iterations(ert, args) -> None:
    if args.num_iterations is not None:
        ert.analysisConfig().set_num_iterations(int(args.num_iterations))
    return ert.analysisConfig().num_iterations
