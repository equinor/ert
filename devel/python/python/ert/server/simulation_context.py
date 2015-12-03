from ert.enkf import ENKF_LIB
from ert.enkf.ert_run_context import ErtRunContext
from ert.enkf.run_arg import RunArg
from ert.job_queue import JobQueueManager
from ert.util import BoolVector, ArgPack, CThreadPool


class SimulationContext(object):
    def __init__(self, ert, size):
        self._ert = ert
        """ :type: ert.enkf.EnKFMain """
        self._size = size

        self._queue_manager = JobQueueManager(ert.siteConfig().getJobQueue())
        self._queue_manager.startQueue(size, verbose=True)

        mask = BoolVector(default_value=True, initial_size=size)
        runpath_fmt = self._ert.getModelConfig().getRunpathFormat()
        subst_list = self._ert.getDataKW()
        self._runpath_list = ErtRunContext.createRunpathList(mask, runpath_fmt, subst_list)

        self._run_args = {}
        """ :type: dict[int, RunArg] """

        self._thread_pool = CThreadPool(8)
        self._thread_pool.addTaskFunction("submitJob", ENKF_LIB, "enkf_main_isubmit_job__")


    def addSimulation(self, iens, target_fs):
        if iens >= self._size:
            raise UserWarning("Realization number out of range: %d >= %d" % (iens, self._size))

        if iens in self._run_args:
            raise UserWarning("Realization number: '%d' already queued" % iens)

        run_arg = RunArg.createEnsembleExperimentRunArg(target_fs, iens, self._runpath_list[iens])
        self._run_args[iens] = run_arg
        self._thread_pool.submitJob(ArgPack(self._ert, run_arg))


    def isRunning(self):
        return self._queue_manager.isRunning()


    def getNumRunning(self):
        return self._queue_manager.getNumRunning()


    def getNumSuccess(self):
        return self._queue_manager.getNumSuccess()


    def getNumFailed(self):
        return self._queue_manager.getNumFailed()


    def didRealizationSucceed(self, iens):
        queue_index = self._run_args[iens].getQueueIndex()
        return self._queue_manager.didJobSucceed(queue_index)


    def didRealizationFail(self, iens):
        queue_index = self._run_args[iens].getQueueIndex()
        return self._queue_manager.didJobFail(queue_index)


    def isRealizationQueued(self, iens):
        return iens in self._run_args


    def isRealizationFinished(self, iens):
        run_arg = self._run_args[iens]

        if run_arg.isSubmitted():
            queue_index = run_arg.getQueueIndex()
            return self._queue_manager.isJobComplete(queue_index)
        else:
            return False
