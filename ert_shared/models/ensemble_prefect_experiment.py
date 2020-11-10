import os
import yaml
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
import time
from ert_shared.ensemble_evaluator.config import load_config
import uuid


class EnsemblePrefectExperiment:
    def __init__(self):
        self._start_time = time.time()
        self._stop_time = None
        self.initial_realizations_mask = []
        self.completed_realizations_mask = []
        self.support_restart = False
        self._queue_running = True
        pass

    @classmethod
    def name(cls):
        return "Ensemble Prefect Experiment"

    def reset(self):
        pass

    def startSimulations(self, arg):
        with open(arg["config_file"], "r") as f:
            config = yaml.safe_load(f)
        ensemble = PrefectEnsemble(config)

        ee_config = load_config()
        ee_id = str(uuid.uuid1()).split("-")[0]
        ee = EnsembleEvaluator(ensemble, ee_config, ee_id=ee_id)
        ee.run_and_get_successful_realizations()
        self._stop_time = time.time()
        self._queue_running = False
        ee.stop()

    def get_runtime(self):
        if self.isFinished():
            return self._stop_time - self._start_time
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
        return {}

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
