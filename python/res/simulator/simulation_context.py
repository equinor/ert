from res.job_queue import JobQueueManager, ForwardModelStatus
from res.enkf import ErtRunContext, EnkfSimulationRunner
from res.enkf.enums import EnkfRunType, HookRuntime
from threading import Thread
from time import sleep


class SimulationContext(object):
    def __init__(self, ert, sim_fs, mask, itr, case_data):
        self._ert = ert
        """ :type: res.enkf.EnKFMain """
        max_runtime = ert.analysisConfig().get_max_runtime()
        self._mask = mask

        job_queue = ert.get_queue_config().create_job_queue()
        job_queue.set_max_job_duration(max_runtime)
        self._queue_manager = JobQueueManager(job_queue)

        subst_list = self._ert.getDataKW()
        path_fmt = self._ert.getModelConfig().getRunpathFormat()
        jobname_fmt = self._ert.getModelConfig().getJobnameFormat()

        self._run_context = ErtRunContext(
            EnkfRunType.ENSEMBLE_EXPERIMENT,
            sim_fs,
            None,
            mask,
            path_fmt,
            jobname_fmt,
            subst_list,
            itr,
        )
        # fill in the missing geo_id data
        for sim_id, (geo_id, _) in enumerate(case_data):
            if mask[sim_id]:
                run_arg = self._run_context[sim_id]
                run_arg.geo_id = geo_id

        self._ert.getEnkfSimulationRunner().createRunPath(self._run_context)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, self._ert)
        self._sim_thread = self._run_simulations_simple_step()

        # Wait until the queue is active before we finish the creation
        # to ensure sane job status while running
        while self.isRunning() and not self._queue_manager.isRunning():
            sleep(0.1)

    def get_run_args(self, iens):
        """
        raises an  exception if no iens simulation found

        :param iens: realization number
        :return: run_args for the realization
        """
        for run_arg in self._run_context:
            if run_arg is not None and run_arg.iens == iens:
                return run_arg
        raise KeyError("No such simulation: %s" % iens)

    def _run_simulations_simple_step(self):
        sim_thread = Thread(
            target=lambda: self._ert.getEnkfSimulationRunner().runSimpleStep(
                self._queue_manager.queue, self._run_context
            )
        )
        sim_thread.start()
        return sim_thread

    def __len__(self):
        return self._mask.count()

    def isRunning(self):
        # TODO: Should separate between running jobs and having loaded all data
        return self._sim_thread.is_alive() or self._queue_manager.isRunning()

    def getNumPending(self):
        return self._queue_manager.getNumPending()

    def getNumRunning(self):
        return self._queue_manager.getNumRunning()

    def getNumSuccess(self):
        return self._queue_manager.getNumSuccess()

    def getNumFailed(self):
        return self._queue_manager.getNumFailed()

    def getNumWaiting(self):
        return self._queue_manager.getNumWaiting()

    def didRealizationSucceed(self, iens):
        queue_index = self.get_run_args(iens).getQueueIndex()
        return self._queue_manager.didJobSucceed(queue_index)

    def didRealizationFail(self, iens):
        # For the purposes of this class, a failure should be anything (killed job, etc) that is not an explicit success.
        return not self.didRealizationSucceed(iens)

    def isRealizationQueued(self, iens):
        # an exception will be raised if it's not queued
        self.get_run_args(iens)
        return True

    def isRealizationFinished(self, iens):
        run_arg = self.get_run_args(iens)

        if run_arg.isSubmitted():
            queue_index = run_arg.getQueueIndex()
            return self._queue_manager.isJobComplete(queue_index)
        else:
            return False

    def __repr__(self):
        running = "running" if self.isRunning() else "not running"
        numRunn = self.getNumRunning()
        numSucc = self.getNumSuccess()
        numFail = self.getNumFailed()
        numWait = self.getNumWaiting()
        fmt = "%s, #running = %d, #success = %d, #failed = %d, #waiting = %d"
        fmt = fmt % (running, numRunn, numSucc, numFail, numWait)
        return "SimulationContext(%s)" % fmt

    def get_sim_fs(self):
        return self._run_context.get_sim_fs()

    def get_run_context(self):
        return self._run_context

    def stop(self):
        self._queue_manager.stop_queue()
        self._sim_thread.join()

    def job_progress(self, iens):
        """Will return a detailed progress of the job.

        The progress report is obtained by reading a file from the filesystem,
        that file is typically created by another process running on another
        machine, and reading might fail due to NFS issues, simultanoues write
        and so on. If loading valid json fails the function will sleep 0.10
        seconds and retry - eventually giving up and returning None. Also for
        jobs which have not yet started the method will return None.

        When the method succeeds in reading the progress file from the file
        system the return value will be an object with properties like this:|

           progress.start_time
           progress.end_time
           progress.run_id
           progress.jobs =[ (job1.name, job1.start_time, job1.end_time, job1.status, job1.error_msg),
                             (job2.name, job2.start_time, job2.end_time, job2.status, job2.error_msg),
                              ....
                             (jobN.name, jobN.start_time, jobN.end_time, jobN.status, jobN.error_msg) ]

        """
        run_arg = self.get_run_args(iens)

        try:
            # will throw if not yet submitted (is in a limbo state)
            queue_index = run_arg.getQueueIndex()
        except ValueError:
            return None
        if self._queue_manager.isJobWaiting(queue_index):
            return None

        return ForwardModelStatus.load(run_arg.runpath)

    def run_path(self, iens):
        """
        Will return the path to the simulation.
        """
        return self.get_run_args(iens).runpath

    def job_status(self, iens):
        """Will query the queue system for the status of the job."""
        run_arg = self.get_run_args(iens)
        try:
            queue_index = run_arg.getQueueIndex()
        except ValueError:
            return None
        return self._queue_manager.getJobStatus(queue_index)
