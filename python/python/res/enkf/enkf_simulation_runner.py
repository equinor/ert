from cwrap import BaseCClass
from ecl.util.util import BoolVector
from res.enkf import EnkfFs
from res.enkf import EnkfPrototype, ErtRunContext
from res.enkf.enums import EnkfInitModeEnum


class EnkfSimulationRunner(BaseCClass):
    TYPE_NAME = "enkf_simulation_runner"

    _create_run_path = EnkfPrototype("bool enkf_main_create_run_path(enkf_simulation_runner, ert_run_context)")
    _run_simple_step = EnkfPrototype("int enkf_main_run_simple_step(enkf_simulation_runner, job_queue, ert_run_context)")

    def __init__(self, enkf_main):
        assert isinstance(enkf_main, BaseCClass)
        # enkf_main should be an EnKFMain, get the _RealEnKFMain object
        real_enkf_main = enkf_main.parent()
        super(EnkfSimulationRunner, self).__init__(
            real_enkf_main.from_param(real_enkf_main).value ,
            parent=real_enkf_main ,
            is_reference=True)

    def _enkf_main(self):
        return self.parent()

    def runSimpleStep(self, job_queue, run_context):
        """ @rtype: int """
        return self._run_simple_step(job_queue, run_context )

    def createRunPath(self, run_context):
        """ @rtype: bool """
        return self._create_run_path(run_context)

    def runEnsembleExperiment(self, job_queue, run_context):
        """ @rtype: int """
        return self.runSimpleStep(job_queue, run_context)


    def runWorkflows(self , runtime):
        """:type res.enkf.enum.HookRuntimeEnum"""
        hook_manager = self._enkf_main().getHookManager()
        hook_manager.runWorkflows(runtime  , self._enkf_main())
