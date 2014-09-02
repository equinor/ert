from functools import partial
from threading import Thread
from ert.job_queue import Workflow

class WorkflowRunner(object):
    def __init__(self, workflow):
        """ @type workflow: Workflow """
        super(WorkflowRunner, self).__init__()

        self.__workflow = workflow
        self.__ert = None

    def run(self, ert, context=None):
        self.__ert = ert

        workflow_thread = Thread(name="ert_gui_workflow_thread")
        workflow_thread.setDaemon(True)
        workflow_thread.run = partial(self.__workflow.run, ert, context=context)
        workflow_thread.start()

    def isRunning(self):
        """ @rtype: bool """
        return self.__workflow.isRunning()

    def isCancelled(self):
        """ @rtype: bool """
        return self.__workflow.isCancelled()

    def cancel(self):
        if self.isRunning():
            self.__workflow.cancel()

    def wait(self):
        self.__workflow.wait()


