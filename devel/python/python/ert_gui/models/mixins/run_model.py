from ert_gui.models.mixins import ModelMixin, AbstractMethodError


class RunModelMixin(ModelMixin):
    RUN_PHASE_CHANGED_EVENT = "run_phase_changed_event"
    RUN_FAILED_EVENT = "run_failed_event"

    def registerDefaultEvents(self):
        super(RunModelMixin, self).registerDefaultEvents()
        self.observable().addEvent(RunModelMixin.RUN_PHASE_CHANGED_EVENT)
        self.observable().addEvent(RunModelMixin.RUN_FAILED_EVENT)


    def __init__(self, phase_count=1, *args):
        super(RunModelMixin, self).__init__(*args)
        self.__phase = 0
        self.__phase_count = phase_count

    def startSimulations(self):
        raise AbstractMethodError(self, "startSimulations")

    def killAllSimulations(self):
        raise AbstractMethodError(self, "killAllSimulations")

    def phaseCount(self):
        return self.__phase_count

    def currentPhase(self):
        return self.__phase

    def allPhasesCompleted(self):
        return self.__phase == self.__phase_count

    def setPhase(self, phase):
        if not 0 <= phase <= self.__phase_count:
            raise ValueError("Phase must be an integer from 0 to less than %d." % self.__phase_count)
        self.__phase = phase
        self.observable().notify(RunModelMixin.RUN_PHASE_CHANGED_EVENT)

    def runFailed(self, message=""):
        self.observable().notify(RunModelMixin.RUN_FAILED_EVENT, message)
