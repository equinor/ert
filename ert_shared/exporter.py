#  Copyright (C) 2020  Equinor ASA, Norway.
#
#  The file 'exporter.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import sys
import logging
from ert_shared import ERT


class Exporter:
    def __init__(self):
        self._export_job = "CSV_EXPORT2"
        self._runpath_job = "EXPORT_RUNPATH"

    def is_valid(self):
        export_job = ERT.enkf_facade.get_workflow_job(self._export_job)
        runpath_job = ERT.enkf_facade.get_workflow_job(self._runpath_job)

        if export_job is None:
            error = "Export not available due to {job_name} is not installed.".format(
                job_name=self._export_job
            )
            logging.warning(error)
            return False

        if runpath_job is None:
            error = "Export not available due to {job_name} is not installed.".format(
                job_name=self._runpath_job
            )
            logging.warning(error)
            return False

        return True

    def run_export(self, parameters):

        logger = logging.getLogger()
        export_job = ERT.enkf_facade.get_workflow_job(self._export_job)
        runpath_job = ERT.enkf_facade.get_workflow_job(self._runpath_job)

        runpath_job.run(ert=ERT.ert, arguments=[], verbose=True)
        if runpath_job.hasFailed():
            raise UserWarning(
                "Failed to execute {job_name}".format(job_name=self._runpath_job)
            )

        export_job.run(
            ert=ERT.ert,
            arguments=[
                ERT.ert.getRunpathList().getExportFile(),
                parameters["output_file"],
                parameters["time_index"],
                parameters["column_keys"],
            ],
            verbose=True,
        )
        if export_job.hasFailed():
            raise UserWarning(
                "Failed to execute {job_name}".format(job_name=self._export_job)
            )
