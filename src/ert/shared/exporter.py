import logging

from ert.libres_facade import LibresFacade

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(self, ert):
        self.ert = ert
        self.facade = LibresFacade(ert)
        self._export_job = "CSV_EXPORT2"
        self._runpath_job = "EXPORT_RUNPATH"

    def is_valid(self):
        export_job = self.facade.get_workflow_job(self._export_job)
        runpath_job = self.facade.get_workflow_job(self._runpath_job)

        if export_job is None:
            logger.error(
                f"Export not available because {self._export_job} is not installed."
            )
            return False

        if runpath_job is None:
            logger.error(
                f"Export not available because {self._runpath_job} is not installed."
            )
            return False

        return True

    def run_export(self, parameters):
        export_job = self.facade.get_workflow_job(self._export_job)
        runpath_job = self.facade.get_workflow_job(self._runpath_job)

        runpath_job.run(ert=self.ert, arguments=[], verbose=True)
        if runpath_job.hasFailed():
            raise UserWarning(f"Failed to execute {self._runpath_job}")

        export_job.run(
            ert=self.ert,
            arguments=[
                str(self.ert.runpath_list_filename),
                parameters["output_file"],
                parameters["time_index"],
                parameters["column_keys"],
            ],
            verbose=True,
        )
        if export_job.hasFailed():
            raise UserWarning(f"Failed to execute {self._export_job}")
