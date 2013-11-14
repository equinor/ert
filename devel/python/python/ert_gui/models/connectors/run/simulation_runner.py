from threading import Thread
import time
from ert_gui.models.connectors.init.case_list import CaseList
from ert_gui.models.connectors.run.run_status import RunStatusModel
from ert_gui.models.mixins import ModelMixin, RunModelMixin


class SimulationRunner(Thread, ModelMixin):
    SIMULATION_PHASE_CHANGED_EVENT = "simulation_phase_changed_event"
    SIMULATION_FINISHED_EVENT = "simulation_finished_event"

    def __init__(self, run_model):
        super(SimulationRunner, self).__init__(name="enkf_main_run_thread")
        self.setDaemon(True)

        assert isinstance(run_model, RunModelMixin)

        run_model.observable().attach(RunModelMixin.RUN_PHASE_CHANGED_EVENT, self.phaseChanged)

        self.__model = run_model


        self.__job_start_time  = 0
        self.__job_stop_time = 0

    def registerDefaultEvents(self):
        super(SimulationRunner, self).registerDefaultEvents()
        self.observable().addEvent(SimulationRunner.SIMULATION_FINISHED_EVENT)
        self.observable().addEvent(SimulationRunner.SIMULATION_PHASE_CHANGED_EVENT)


    def startStatusThread(self):
        status_thread = Thread(name="enkf_main_run_status_poll_thread")
        status_thread.setDaemon(True)
        status_thread.run = RunStatusModel().startStatusPoller
        status_thread.start()
        return status_thread

    def run(self):
        self.__job_start_time = int(time.time())

        self.startStatusThread()

        self.__model.startSimulations()
        self.__job_stop_time = int(time.time())

        CaseList().externalModificationNotification()

        self.observable().notify(SimulationRunner.SIMULATION_FINISHED_EVENT)

    def getRunningTime(self):
        if self.__job_stop_time < self.__job_start_time:
            return time.time() - self.__job_start_time
        else:
            return self.__job_stop_time - self.__job_start_time

    def killAllSimulations(self):
        self.__model.killAllSimulations()


    def getTotalProgress(self):
        phase = self.__model.currentPhase()
        phase_count = self.__model.phaseCount()

        return phase, phase_count


    def phaseChanged(self):
        self.observable().notify(SimulationRunner.SIMULATION_PHASE_CHANGED_EVENT)

        phase, phase_count = self.getTotalProgress()
        if phase < phase_count:
            RunStatusModel().waitUntilFinished()
            self.startStatusThread()






