from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError

class SingleTestRun(BaseRunModel):
    
    def __init__(self):
        super(SingleTestRun, self).__init__("Single test-run")

    def runSimulations(self, job_queue,  arguments):
        pass
