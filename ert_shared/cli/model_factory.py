from argparse import ArgumentTypeError
import os

from ecl.util.util import BoolVector
from ert_shared.ide.keywords.definitions import RangeStringArgument
from ert_shared import ERT
from ert_shared.models.ensemble_experiment import EnsembleExperiment
from ert_shared.models.ensemble_smoother import EnsembleSmoother
from ert_shared.models.iterated_ensemble_smoother import IteratedEnsembleSmoother
from ert_shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert_shared.models.single_test_run import SingleTestRun


def create_model(args):
    # Setup model
    if args.mode == "test_run":
        model, argument = _setup_single_test_run()
    elif args.mode == "ensemble_experiment":
        model, argument = _setup_ensemble_experiment(args)
    elif args.mode == "ensemble_smoother":
        model, argument = _setup_ensemble_smoother(args)
    elif args.mode == "es_mda":
        model, argument = _setup_multiple_data_assimilation(args)
    elif args.mode == "iterative_ensemble_smoother":
        model, argument = _setup_iterative_ensemble_smoother(args)

    else:
        raise NotImplementedError("Run type not supported {}".format(args.mode))

    return model, argument


def _setup_single_test_run():
    model = SingleTestRun()
    simulations_argument = {
        "active_realizations": BoolVector(default_value=True, initial_size=1),
    }
    return model, simulations_argument


def _setup_ensemble_experiment(args):
    model = EnsembleExperiment()
    simulations_argument = {
        "active_realizations": _realizations(args),
        "iter_num": int(args.iter_num),
    }
    return model, simulations_argument


def _setup_ensemble_smoother(args):
    model = EnsembleSmoother()
    iterable = False
    active_name = ERT.ert.analysisConfig().activeModuleName()
    modules = ERT.enkf_facade.get_analysis_module_names(iterable=iterable)
    simulations_argument = {
        "active_realizations": _realizations(args),
        "target_case": _target_case_name(args, format_mode=False),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
    }
    return model, simulations_argument


def _setup_multiple_data_assimilation(args):
    model = MultipleDataAssimilation()
    iterable = False
    active_name = ERT.ert.analysisConfig().activeModuleName()
    modules = ERT.enkf_facade.get_analysis_module_names(iterable=iterable)
    simulations_argument = {
        "active_realizations": _realizations(args),
        "target_case": _target_case_name(args, format_mode=True),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
        "weights": args.weights,
        "start_iteration": int(args.start_iteration),
    }
    return model, simulations_argument


def _setup_iterative_ensemble_smoother(args):
    model = IteratedEnsembleSmoother()
    iterable = True
    active_name = ERT.ert.analysisConfig().activeModuleName()
    modules = ERT.enkf_facade.get_analysis_module_names(iterable=iterable)
    simulations_argument = {
        "active_realizations": _realizations(args),
        "target_case": _target_case_name(args, format_mode=True),
        "analysis_module": _get_analysis_module_name(
            active_name, modules, iterable=iterable
        ),
        "num_iterations": _num_iterations(args),
    }
    return model, simulations_argument


def _get_analysis_module_name(active_name, modules, iterable):

    if active_name in modules:
        return active_name
    elif "STD_ENKF" in modules and not iterable:
        return "STD_ENKF"
    elif "RML_ENKF" in modules and iterable:
        return "RML_ENKF"
    elif len(modules) > 0:
        return modules[0]

    return None


def _realizations(args):
    ensemble_size = ERT.enkf_facade.get_ensemble_size()
    mask = BoolVector(default_value=False, initial_size=ensemble_size)
    if args.realizations is None:
        default = "0-{}".format(ensemble_size - 1)
        mask.updateActiveMask(default)
        return mask

    validator = RangeStringArgument(ensemble_size)
    validated = validator.validate(args.realizations)
    if validated.failed():
        raise ArgumentTypeError(
            "Defined realizations is not within range of ensemble size: {}".format(
                args.realizations
            )
        )
    mask.updateActiveMask(args.realizations)
    return mask


def _target_case_name(args, format_mode=False):
    """@rtype: str"""
    if args.target_case is not None:
        return args.target_case

    if not format_mode:
        case_name = ERT.enkf_facade.get_current_case_name()
        return "{}_smoother_update".format(case_name)

    aic = ERT.ert.analysisConfig().getAnalysisIterConfig()
    if aic.caseFormatSet():
        return aic.getCaseFormat()

    case_name = ERT.enkf_facade.get_current_case_name()
    return "{}_%d".format(case_name)


def _num_iterations(args):
    if args.num_iterations is not None:
        ERT.ert.analysisConfig().getAnalysisIterConfig().setNumIterations(
            int(args.num_iterations)
        )
    return None
