import os
import yaml
from ert_shared.ensemble_evaluator.entity.prefect_ensamble import PrefectEnsemble, SharedDiskStorageDriver
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
import time
from ert_shared.ensemble_evaluator.config import load_config
import uuid


class EnsemblePrefectExperiment:
    def __init__(self):
        self._start_time = time.time()
        self.initial_realizations_mask = []
        self.completed_realizations_mask = []
        self.support_restart = False
        self._queue_running = False
        pass

    @classmethod
    def name(cls):
        return "Ensemble Prefect Experiment"

    def reset(self):
        pass

    def startSimulations(self, arg):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        ensemble = PrefectEnsemble(config)

        ee_config = load_config()
        ee_id = str(uuid.uuid1()).split("-")[0]
        ee = EnsembleEvaluator(ensemble, ee_config, ee_id=ee_id)
        self._queue_running = True
        ee.run()
        self._queue_running = False
        ee.stop()

    def get_runtime(self):
        return time.time() - self._start_time

    def currentPhase(self):
        return 1 if self.isFinished() else 0

    def isQueueRunning(self):
        return self._queue_running

    def getPhaseName(self):
        return "0"

    def phaseCount(self):
        return 2

    def getQueueStatus(self):
        """ @rtype: dict of (JobStatusType, int) """
        queue_status = {}

        # for job_number in range(len(self._job_queue)):
        #     status = self._job_queue.getJobStatus(job_number)
        #
        #     if not status in queue_status:
        #         queue_status[status] = 0
        #
        #     queue_status[status] += 1

        return queue_status

    def getQueueSize(self):
        """ @rtype: int """
        return 1

    def isFinished(self):
        return not self._queue_running

    def isIndeterminate(self):
        """ @rtype: bool """
        return False

    def getDetailedProgress(self):
        return {}, -1

    def killAllSimulations(self):
        print("Kill all the simulations")

    def hasRunFailed(self):
        return False

    def getFailMessage(self):
        return ""