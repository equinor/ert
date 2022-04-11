from typing import List

from ert.ensemble_evaluator.activerange import ActiveRange
from ert_shared.models.ensemble_experiment import EnsembleExperiment
from ert_shared.models.ensemble_smoother import EnsembleSmoother
from ert_shared.models.iterated_ensemble_smoother import IteratedEnsembleSmoother
from ert_shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert_shared.models.single_test_run import SingleTestRun


def create_model(ert, get_analysis_modules, ensemble_size, current_case_name, args):
    # Setup model
    if args.mode == "test_run":
        model = _setup_single_test_run(ert)
    elif args.mode == "ensemble_experiment":
        model = _setup_ensemble_experiment(ert, args, ensemble_size)
    elif args.mode == "ensemble_smoother":
        model = _setup_ensemble_smoother(
            ert, get_analysis_modules, args, ensemble_size, current_case_name
        )
    elif args.mode == "es_mda":
        model = _setup_multiple_data_assimilation(
            ert, get_analysis_modules, args, ensemble_size, current_case_name
        )
    elif args.mode == "iterative_ensemble_smoother":
        model = _setup_iterative_ensemble_smoother(
            ert, get_analysis_modules, args, ensemble_size, current_case_name
        )

    else:
        raise NotImplementedError("Run type not supported {}".format(args.mode))

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


def _setup_ensemble_smoother(
    ert, get_analysis_modules, args, ensemble_size, current_case_name
):
    iterable = False
    modules = get_analysis_modules(iterable=iterable)
    active_name = ert.analysisConfig().activeModuleName()
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=False
        ),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
    }
    model = EnsembleSmoother(simulations_argument, ert, ert.get_queue_config())
    return model


def _setup_multiple_data_assimilation(
    ert, get_analysis_modules, args, ensemble_size, current_case_name
):
    iterable = False
    active_name = ert.analysisConfig().activeModuleName()
    modules = get_analysis_modules(iterable=iterable)
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=True
        ),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
        "weights": args.weights,
        "start_iteration": int(args.start_iteration),
    }
    model = MultipleDataAssimilation(simulations_argument, ert, ert.get_queue_config())
    return model


def _setup_iterative_ensemble_smoother(
    ert, get_analysis_modules, args, ensemble_size, current_case_name
):
    iterable = True
    active_name = ert.analysisConfig().activeModuleName()
    modules = get_analysis_modules(iterable=iterable)
    simulations_argument = {
        "active_realizations": _realizations(args, ensemble_size),
        "target_case": _target_case_name(
            ert, args, current_case_name, format_mode=True
        ),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
        "num_iterations": _num_iterations(ert, args),
    }
    model = IteratedEnsembleSmoother(simulations_argument, ert, ert.get_queue_config())
    return model


def _get_analysis_module_name(active_name, modules, iterable):

    if active_name in modules:
        return active_name
    elif "STD_ENKF" in modules and not iterable:
        return "STD_ENKF"
    elif "LIB_IES" in modules and iterable:
        return "LIB_IES"
    elif len(modules) > 0:
        return modules[0]

    return None


def _realizations(args, ensemble_size: int) -> List[bool]:
    if args.realizations is None:
        return [True] * ensemble_size
    return ActiveRange(rangestring=args.realizations, length=ensemble_size).mask


def _target_case_name(ert, args, current_case_name, format_mode=False):
    """@rtype: str"""
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        return "{}_smoother_update".format(current_case_name)

    aic = ert.analysisConfig().getAnalysisIterConfig()
    if aic.caseFormatSet():
        return aic.getCaseFormat()

    return "{}_%d".format(current_case_name)


def _num_iterations(ert, args):
    if args.num_iterations is not None:
        ert.analysisConfig().getAnalysisIterConfig().setNumIterations(
            int(args.num_iterations)
        )
    return None
