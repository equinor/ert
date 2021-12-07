#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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
import os.path
import pathlib
from res.enkf import EnKFMain, ResConfig
from distutils.dir_util import copy_tree
import tempfile
import shutil


class ErtTestContext:
    def __init__(self, test_name, model_config):
        self._tmp_dir = tempfile.mkdtemp()
        self._model_config = model_config
        self._res_config = None
        self._ert = None
        self._dir_before = None

    def __enter__(self):
        self._dir_before = os.getcwd()
        os.chdir(self._tmp_dir)
        try:
            dir = pathlib.Path(self._model_config).parent
            config = pathlib.Path(self._model_config).name
            copy_tree(dir, self._tmp_dir)
            self._res_config = ResConfig(user_config_file=config)
            self._ert = EnKFMain(self._res_config, strict=True)
        except Exception:
            os.chdir(self._dir_before)
            shutil.rmtree(self._tmp_dir)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ert = None
        self._res_config = None
        os.chdir(self._dir_before)
        shutil.rmtree(self._tmp_dir)

    def getErt(self):
        return self._ert

    def getCwd(self):
        return os.getcwd()

    def installWorkflowJob(self, job_name, job_path):
        """@rtype: bool"""
        if os.path.exists(job_path) and os.path.isfile(job_path):
            ert = self.getErt()
            workflow_list = ert.getWorkflowList()

            workflow_list.addJob(job_name, job_path)
        else:
            raise IOError("Could not load workflowjob from:%s" % job_path)

    def runWorkflowJob(self, job_name, *arguments):
        """@rtype: bool"""
        ert = self.getErt()
        workflow_list = ert.getWorkflowList()

        if workflow_list.hasJob(job_name):
            job = workflow_list.getJob(job_name)
            job.run(ert, [arg for arg in arguments])
            return True
        else:
            return False
