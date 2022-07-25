from typing import List

from ert_shared.models.ensemble_experiment import EnsembleExperiment
from ert_shared.models.ensemble_smoother import EnsembleSmoother
from ert_shared.models.iterated_ensemble_smoother import IteratedEnsembleSmoother
from ert_shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert_shared.models.single_test_run import SingleTestRun
from res.config.active_range import ActiveRange


def create_model(ert, ensemble_size, current_case_name, args):
    # Setup model
    if args.mode == "test_run":
        model = _setup_single_test_run(ert)
    elif args.mode == "ensemble_experiment":
        model = _setup_ensemble_experiment(ert, args, ensemble_size)
    elif args.mode == "ensemble_smoother":
        model = _setup_ensemble_smoother(ert, args, ensemble_size, current_case_name)
    elif args.mode == "es_mda":
        model = _setup_multiple_data_assimilation(
            ert, args, ensemble_size, current_case_name
        )
    elif args.mode == "iterative_ensemble_smoother":
        model = _setup_iterative_ensemble_smoother(
            ert, args, ensemble_size, current_case_name
        )

    else:
        raise NotImplementedError(f"Run type not supported {args.mode}")

    return model


def _setup_single_test_run(ert):
    simulations_argument = {"active_realizations": [True]}
    model = SingleTestRun(simulations_argument, ert)
    return model


def _setup_ensemble_experiment(ert, args, ensemble_size):
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "iter_num": int(args.iter_num),
    }
    model = EnsembleExperiment(simulations_argument, ert, ert.get_queue_config())
    return model


def _setup_ensemble_smoother(ert, args, ensemble_size, current_case_name):
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=False
        ),
        "analysis_module": "STD_ENKF",
    }
    model = EnsembleSmoother(simulations_argument, ert, ert.get_queue_config())
    return model


def _setup_multiple_data_assimilation(ert, args, ensemble_size, current_case_name):
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=True
        ),
        "analysis_module": "STD_ENKF",
        "weights": args.weights,
        "start_iteration": int(args.start_iteration),
    }
    model = MultipleDataAssimilation(simulations_argument, ert, ert.get_queue_config())
    return model


def _setup_iterative_ensemble_smoother(ert, args, ensemble_size, current_case_name):
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=True
        ),
        "analysis_module": "IES_ENKF",
        "num_iterations": _num_iterations(ert, args),
    }
    model = IteratedEnsembleSmoother(simulations_argument, ert, ert.get_queue_config())
    return model


def _realizations(args, ensemble_size: int) -> List[bool]:
    if args.realizations is None:
        return [True] * ensemble_size
    return ActiveRange(rangestring=args.realizations, length=ensemble_size).mask


def _target_case_name(ert, args, current_case_name, format_mode=False):
    """@rtype: str"""
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        return f"{current_case_name}_smoother_update"

    aic = ert.analysisConfig().getAnalysisIterConfig()
    if aic.caseFormatSet():
        return aic.getCaseFormat()

    return f"{current_case_name}_%d"


def _num_iterations(ert, args) -> None:
    if args.num_iterations is not None:
        ert.analysisConfig().getAnalysisIterConfig().setNumIterations(
            int(args.num_iterations)
        )
