from cwrap import BaseCClass
from res.enkf import EnkfFs
from res.enkf import EnkfPrototype, ErtRunContext
from res.enkf.enums import EnkfInitModeEnum
from ecl.util import BoolVector


class EnkfSimulationRunner(BaseCClass):
    TYPE_NAME = "enkf_simulation_runner"

    _create_run_path = EnkfPrototype("bool enkf_main_create_run_path(enkf_simulation_runner, ert_run_context)")
    _run_simple_step = EnkfPrototype("int enkf_main_run_simple_step(enkf_simulation_runner, job_queue, bool_vector, enkf_init_mode_enum, int)")

    def __init__(self, enkf_main):
        assert isinstance(enkf_main, BaseCClass)
        super(EnkfSimulationRunner, self).__init__(enkf_main.from_param(enkf_main).value, parent=enkf_main, is_reference=True)
        self.ert = enkf_main
        """:type: res.enkf.EnKFMain """

    def runSimpleStep(self, job_queue, run_context):
        """ @rtype: int """
        return self._run_simple_step(job_queue, run_context.get_mask( ), initialization_mode , run_context.get_iter( ))

    def createRunPath(self, run_context):
        """ @rtype: bool """
        return self._create_run_path(run_context)

    def runEnsembleExperiment(self, job_queue, run_context):
        """ @rtype: int """
        return self.runSimpleStep(job_queue, run_context)


    def runWorkflows(self , runtime):
        """:type res.enkf.enum.HookRuntimeEnum"""
        hook_manager = self.ert.getHookManager()
        hook_manager.runWorkflows( runtime  , self.ert )
