from concurrent import futures
from typing import TYPE_CHECKING, Optional

from ert._c_wrappers.job_queue import Workflow

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain


class WorkflowRunner:
    def __init__(
        self,
        workflow: Workflow,
        ert: Optional["EnKFMain"] = None,
    ):
        super().__init__()

        self.__workflow = workflow
        self.__ert = ert

        self.__workflow_result = None
        self._workflow_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._workflow_future: futures.Future = None

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, _type, value, traceback):
        self.wait()

    def run(self):
        if self._workflow_future and self._workflow_future.running():
            raise AssertionError("An instance of workflow is already running!")
        self._workflow_future = self._workflow_executor.submit(
            self.__workflow.run, self.__ert
        )

    @property
    def workflow_status(self):
        return self.__workflow.get_status()

    def cancel(self):
        if self.workflow_status.is_running():
            self.__workflow.cancel()
        self.wait()

    def exception(self):
        if self._workflow_future is not None:
            return self._workflow_future._exception
        return None

    def wait(self):
        # This returns a tuple (done, pending), since we run only one job we don't
        # need to use it
        _, _ = futures.wait(
            [self._workflow_future], timeout=None, return_when=futures.FIRST_EXCEPTION
        )

    def get_failed_jobs(self):
        return self.__workflow.get_failed_jobs()
