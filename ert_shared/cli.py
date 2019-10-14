#!/usr/bin/env python
import os
from argparse import ArgumentTypeError

from ecl.util.util import BoolVector
from res.enkf import EnKFMain, ResConfig

from ert_shared import ERT
from ert_gui.ide.keywords.definitions import RangeStringArgument
from .models.ensemble_experiment import EnsembleExperiment
from .models.ensemble_smoother import EnsembleSmoother
from .models.multiple_data_assimilation import MultipleDataAssimilation
from .models.single_test_run import SingleTestRun
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_cli(args):

    res_config = ResConfig(args.config)
    os.chdir(res_config.config_path)
    ert = EnKFMain(res_config, strict=True, verbose=args.verbose)
    notifier = ErtCliNotifier(ert, args.config)
    ERT.adapt(notifier)
    
    if args.mode == 'workflow':
        _execute_workflow(args.name)
        return

    # Setup model
    if args.mode == 'test_run':
        model, argument = _setup_single_test_run()
    elif args.mode == 'ensemble_experiment':
        model, argument = _setup_ensemble_experiment(args)
    elif args.mode == 'ensemble_smoother':
        model, argument = _setup_ensemble_smoother(args)
    elif args.mode == 'es_mda':
        model, argument = _setup_multiple_data_assimilation(args)
    
    else:
        raise NotImplementedError(
            "Run type not supported {}".format(args.mode))

    model.runSimulations(argument)

def _execute_workflow(workflow_name):
    workflow_list = ERT.ert.getWorkflowList()
    try:
        workflow = workflow_list[workflow_name]
    except KeyError:
        logging.error("Workflow {} is not in the list of available workflows".format(workflow_name))
        return
    context = workflow_list.getContext()
    workflow.run(ert=ERT.ert, verbose=True, context=context)
    all_successfull = all([v['completed'] for k, v in workflow.getJobsReport().items()])
    if all_successfull:
        logging.info("Workflow {} ran successfully!".format(workflow_name))

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
        "analysis_module": _get_analysis_module_name(active_name, modules, iterable=iterable),
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
        "analysis_module": _get_analysis_module_name(active_name, modules, iterable=iterable),
        "weights": args.weights
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
            "Defined realizations is not within range of ensemble size: {}".format(args.realizations))
    mask.updateActiveMask(args.realizations)
    return mask


def _target_case_name(args, format_mode=False):
    """ @rtype: str """
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


class ErtCliNotifier():
    """CLI Notifier to use in ERT Adapter"""

    def __init__(self, ert, config_file):
        self._ert = ert
        self._config_file = config_file

    @property
    def ert(self):
        """ @rtype: EnKFMain """
        if self._ert is None:
            raise ValueError("Ert is undefined.")
        return self._ert

    @property
    def config_file(self):
        """ @rtype: str """
        if self._ert is None:
            raise ValueError("Ert is undefined.")
        return self._config_file

    @property
    def ertChanged(self):
        pass

    def emitErtChange(self):
        pass

    def reloadERT(self, config_file):
        pass
